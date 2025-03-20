import pickle
import time
import random

import numpy as np
import torch

from drrl.common.utils import setup_logger, compute_gae_reward
from drrl.data.ppo_data_frame import PPODataFrame
from drrl.network.zmq import ZmqAdaptor
from drrl.network.name_client import NameClient
from drrl.network.log_client import LogClient
from drrl.network.inference_client import InferenceClient


# agent负责和环境交互
class MultiAgentPPOTeam():

    def __init__(self, team_id: str, cfg, policy=None, inference=False, local_test=False, dont_send=False):
        # game base info
        self.team_name = team_id
        self.inference = inference
        self.dont_send = dont_send

        if policy is not None:
            self.policy = policy
            self.policy.eval()

        self.logger = setup_logger("agent.log")
        self.logger.info("init gent")

        self.cfg = cfg
        self.player_num = cfg['policy']['placeholder']['max_agents']
        # self.team_num = cfg['policy']['placeholder']['max_teams']
        self.action_space = cfg['policy']['placeholder']['action_space']

        self.n_step = cfg['n_steps']
        self.gamma = cfg['gamma']
        self.lam = cfg['lam']
        self.n_lstm = cfg['n_lstm']

        self.instances = []
        self.cur_lstm_states = np.zeros(shape=(self.player_num, self.n_lstm * 2), dtype=np.float32)

        self.ls_api = None
        if not local_test:
            self.logger.info("Agent init net client and log client")
            self.ns_api = NameClient(cfg)
            self._net = ZmqAdaptor(logger=self.logger)

            self.ls_api = LogClient(cfg, ns_api=self.ns_api, net=self._net)
            self.ls_api.connect()
            self.ls_api.add_moni(msg_type="agent", interval_time=0)

        self.apm = 0
        self.sample_count = 0
        self.predict_timeout = 0
        self.action_time_list = []

        self.ep_step = 0
        self.ep_reward = 0
        self.ep_action = [0] * self.action_space
        self.pre_action = np.zeros((self.player_num, 2))
        self.plenty_reward = np.zeros(self.player_num)
        self.success = 0
        self.out_of_road = 0
        self.crash_vehicle = 0

        if not local_test:
            self.if_api = InferenceClient(cfg, ns_api=self.ns_api, net=self._net)

            while True:
                _, q_services = self.ns_api.discovery_q_server(block=True)
                # q_services = list(filter(lambda x: x.extra["data_type"] == "training" and x.extra['team_id'] == self.team_name, q_services))
                q_services = list(filter(lambda x: x.extra["data_type"] == "training", q_services))
                if len(q_services) == 0:
                    continue

                q = random.choice(q_services)
                self._net.start({"mode": "req", "host": q.address, "port": q.port, "timeout": 2500})
                break
            self.logger.info("Agent connect inference server")

            # net client to train server
            while True:
                train = None
                _, train_services = self.ns_api.discovery_train_server(block=True)
                # train_services = list(filter(lambda x: x.extra["team_id"] == self.team_name, train_services))
                if len(train_services) == 0:
                    continue

                self.logger.info(f"myself team name is {self.team_name} and data server team service is {train_services[0].extra['team_id']}")

                # random choice one train server
                train = random.choice(train_services)
                # train = train_services[0]
                break

            self._net.start({"mode": "push", "host": train.address, "port": train.port})
            self.logger.info("Agent connected inference server")

    def act(self, actor_observations, critic_observations, last_reward, last_done, info=None):
        # moni record
        self.record_delay_time()

        self.ep_step += 1
        if isinstance(last_reward, list) or isinstance(last_reward, np.ndarray):
            last_reward = last_reward[0]

        rewards = np.zeros(self.player_num)
        done = np.zeros(self.player_num)
        current_done = np.zeros(self.player_num)

        if type(last_done) is not bool:
            for i in range(self.player_num):
                if f'agent{i}' not in last_done or last_done[f'agent{i}'] is True:
                    done[i] = 1
                if f'agent{i}' in last_reward:
                    rewards[i] = last_reward[f'agent{i}'] #  + self.plenty_reward[i]

            for k, v in info.items():
                if v['arrive_dest']:
                    self.success += 1
                if v['out_of_road']:
                    self.out_of_road += 1
                if v['crash_vehicle']:
                    self.crash_vehicle += 1
        else:
            rewards[0] = last_reward
            done[0] = last_done

            if info['arrive_dest']:
                self.success += 1
            if info['out_of_road']:
                self.out_of_road += 1
            if info['crash_vehicle']:
                self.crash_vehicle += 1

        self.ep_reward += np.sum(rewards)


        # 新的一轮
        if np.all(done) or self.ep_step == 1:
            current_done = np.zeros(self.player_num) # 新的一轮推理用 current
        else:
            current_done = done
        instance = PPODataFrame(
            actor_obs=critic_observations,
            critic_obs=critic_observations,
            # mask=None,
            lstm_state=self.cur_lstm_states, # dummy
            done=current_done, # 推理时需要判断 done, 如果结束则不计算任何loss
        )

        if self.inference:
            result = self.if_api.remote_predict(instance)
            # instance.done = done # 训练用 last done
            action = self.result_post_processing(result, instance, rewards, current_done)
        else:
            torch.set_num_threads(2)
            obs = torch.tensor(actor_observations, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)
            # pre_action = torch.tensor(self.pre_action, dtype=torch.long)
            action, _, _, _ = self.policy(obs)
            action = action.detach().numpy()[0]

        # for i, v in enumerate(action):
        #     self.ep_action[i] += v

        return action

    def result_post_processing(self, result, instance: PPODataFrame, last_reward, done):
        """
        inference server concat all model output for convenient.
        agent needs to get different variable from result.
        all variable's type in result are float32.

        only rollout data generate through inference server(online policy) will use to training.
        """        # match your action output num
        if self.cfg.algo == 'ppo':
            result = {
                "action": result[:, 0:2], # 返回40个，按需过滤
                "logprob": result[:, 2:4],
                "value": result[:, 4],
                # "state_out": result[:, 3:-1],
                "state_out": [-1], # dummy
                "model_update_times": result[:, -1],
            }

        else: # sac
            result = {
                "action": result[:, 0:2], # 返回40个，按需过滤
                "logprob": result[0, 0], # dummy
                "value": result[0, 0], # dummy
                # "state_out": result[:, 3:-1],
                "state_out": [-1], # dummy
                "model_update_times": result[:, -1],
            }

        action = result["action"].astype('float32')
        value = result["value"]
        state_out = result["state_out"]

        self.cur_lstm_states = state_out

        instance.action = action
        instance.value = value
        instance.logprob = result["logprob"]
        instance.lstm_state = result["state_out"]

        instance.model_update_times = result["model_update_times"]

        if len(self.instances) >= 1:
            self.instances[-1].reward = last_reward
            self.instances[-1].next_obs = instance.actor_obs
            # self.instances[-1].done = done

        self.instances.append(instance)

        if len(self.instances) == self.n_step + 1 and self.dont_send is not True:
            self.send_instance()

        # action_diff = np.abs(action - self.pre_action)
        # self.plenty_reward = np.zeros(self.player_num)
        # for i in range(self.player_num):
        #     if action_diff[i][0] > 1.2 * 3:
        #         self.plenty_reward[i] -= 0.1 * action_diff[i][0]
        #     if action_diff[i][1] > 1.2 * 3:
        #         self.plenty_reward[i] -= 0.1 * action_diff[i][1]
        self.pre_action = action
        post_pocess_action = {}
        for idx, d in enumerate(done):
            if d != 0:
                continue

            if self.cfg.algo == 'ppo':
                post_pocess_action[f'agent{idx}'] = np.clip(action[idx], -3, 3) / 3
            else:
                post_pocess_action[f'agent{idx}'] = action[idx]

        return post_pocess_action

    def send_instance(self):
        """
        send rollout data to training server every n_steps data collected.
        """
        # mb stands for minibatch
        # assert len(self.instances) >= self.n_step, "rollout len need > n_step"
        instances_rollout = self.instances[:self.n_step]

        next_value = self.instances[self.n_step].value
        next_done = self.instances[self.n_step].done
        self.instances = self.instances[self.n_step:]

        instances_rollout = compute_gae_reward(instances_rollout, self.gamma, self.lam, next_value, next_done)

        result = {"instances": instances_rollout}
        result = pickle.dumps(result)
        self._net.sender.send(result)
        # self.logger.info("send instance num {0}".format(len(instances_rollout)))
        # self.ls_api.record(msg_type="agent", data={"env_steps": len(instances_rollout) * self.player_num, "apm": self.apm})
        self.ls_api.record(msg_type="agent", data={"env_steps": len(instances_rollout), "apm": self.apm})
        self.ls_api.send_moni(msg_type="agent")

    def send_end_info(self, end_info):
        """
        runner does not have ls_api, it use agent.send_end_info to send env info to log server.
        """
        if self.ls_api is not None and self.dont_send is not True:
            self.ls_api.record(
                msg_type="agent", data={
                    "predict_timeout": self.predict_timeout,
                })

            end_info["ep_reward"] = self.ep_reward
            end_info["ep_len"] = self.ep_step
            end_info["success"] = self.success
            end_info["out_of_road"] = self.out_of_road
            end_info["crash_vehicle"] = self.crash_vehicle

            for i in range(len(self.ep_action)):
                end_info["z_ep_action_" + str(i)] = self.ep_action[i]

            self.ls_api.record(msg_type="agent", data=end_info)
            self.ls_api.send_moni(msg_type="agent")

    def state_reset(self):
        """
        runner needs to reset agents variable when a game match end.
        """
        # reset
        self.action_time_list = []
        self.cur_lstm_states = np.zeros(shape=(self.player_num, self.n_lstm * 2), dtype=np.float32)

        self.ep_step = 0
        self.ep_action = [0] * self.action_space
        self.ep_reward = 0
        self.success = 0
        self.out_of_road = 0
        self.crash_vehicle = 0

        self.pre_action = np.zeros((self.player_num, 2))
        self.plenty_reward = np.zeros(self.player_num)

    def record_delay_time(self):
        # 毫秒 ms
        tmp = int(round(time.time() * 1000))
        self.action_time_list.append(tmp)
        if len(self.action_time_list) > self.n_step:
            self.action_time_list.pop(0)
            # 操作 n_step 次的时间差计算 apm
            self.apm = (60 * 1000 * self.n_step) / (tmp - self.action_time_list[0])
        elif len(self.action_time_list) == self.n_step:
            self.apm = (60 * 1000 * self.n_step) / (tmp - self.action_time_list[0])

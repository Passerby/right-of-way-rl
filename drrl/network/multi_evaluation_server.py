import os
import pickle
import signal
import time
from enum import Enum

import numpy as np
import zmq
import torch

from drrl.network.zmq import ZmqAdaptor
from drrl.common.utils import setup_logger, is_development, create_path
from drrl.network.name_client import NameClient
from drrl.network.log_client import LogClient
from drrl.network.utils import cal_elo

from drrl.models.policies.multi_team_policy import MultiTeamPolicy


class EvaluationServer:
    """ 模型池服务
    """

    class Req(Enum):
        """ 枚举服务请求类型
        """
        FINISH_MATCH = "finish_match"
        GET_POOL_MODEL = "get_pool_model"
        GET_LATEST_MODEL = "get_latest_model"
        QUERY_TASK = "query_task"
        QUERY_EVAL = "query_eval"
        ALL = "all"

    class Res(Enum):
        """ 枚举返回相应类型
        """
        OK = "ok"
        INVALID_PARAM = "invalid_param"
        INVALID_API = "invalid_api"
        NOT_FOUND_MODEL = "not_found_model"

    def __init__(self, cfg):
        self.cfg = cfg
        self.ns_api = NameClient(cfg)
        self.logger = setup_logger("evaluation_server.log")
        self._nets = [ZmqAdaptor(logger=self.logger), ZmqAdaptor(logger=self.logger)]
        self._net = self._nets[0] # right player -> first_0
        self._net_left = self._nets[1] # left player -> second_0

        # play_mode set in config : build-in-bot, self-play (init)
        self.play_mode = cfg.evaluation_server["play_mode"]

        address, port = self.ns_api.register(rtype="evaluation_server", extra={"data_type": "evaluation", "zmq_mode": "rep"})
        self._net.start({"mode": "rep", "host": "*", "port": port})

        self.latest_model_dict = None

        self.model_pool = {}
        # self.model_result_table = {
        #     "build-in-bot": [],
        #     "training": [],
        # }
        if is_development():
            self.update_model_time = time.time() + 30
        else:
            self.update_model_time = time.time() + 60 * 1

        self.population_n = 10
        self.elo = {"build-in-bot": 1500.0, "training": 1500.0, "second_0": 1500, "first_0": 1500}
        self.elo_new_n = 50
        self.elo_k = 16 # 新模型进行评估的时候应增大到 2k，elo_new_n 场之后减小至原来 k，如果模型池足够大，高端排名应该缩小 k

        self.need_to_eval_models = []
        self.history_eval_resutls = {} # key: (left, right) value: [{rewards: [l_r, r_r], match_result: 1, 0, 0.5},...]
        self.build_in_ep_reward = []
        self.first_ep_reward = []
        self.second_ep_reward = []
        self.first_model_index = {'first_0': 0} # model time -> index
        self.second_model_index = {'second_0': 0} # model time -> index
        self.inference_servers = {} # server_uuid -> model_time

        # sub model
        # net clinet to publish model server
        # TODO: 改成两个 service
        while True:
            _res, pub_serivces = self.ns_api.discovery_pub_model_server(block=True)
            if len(pub_serivces) >= 2:
                filter_first_pub_services = list(filter(lambda x: x.extra['team_id'] == "first_0", pub_serivces))
                self._net.start({"mode": "sub", "host": filter_first_pub_services[0].address, "port": filter_first_pub_services[0].port})
                filter_second_pub_services = list(filter(lambda x: x.extra['team_id'] == "second_0", pub_serivces))
                self._net_left.start({"mode": "sub", "host": filter_second_pub_services[0].address, "port": filter_second_pub_services[0].port})
                break

            self.logger.info("wait for pub model server")
            time.sleep(1)

        # net client to log server
        self.ls_api = LogClient(cfg, ns_api=self.ns_api, net=self._net)
        self.ls_api.connect()
        self.ls_api.add_moni(msg_type="evaluation_server", interval_time=60)

        self.start_time = time.time()
        self.moni_check_time = time.time() + 30
        self.send_wx_next_time = time.time() + 60 * 30
        self.next_dump_time = time.time() + 60 if is_development() else time.time() + 60 * 60
        self.kill_now = False

        self.dump_obs = {
            "first_0": [],
            "second_0": []
        }

        # 读取 dump_obs.pkl 文件，用于计算 KL 散度 = logits 的差异
        with open('dump_obs.pkl', 'rb') as f:
            self.dump_obs = pickle.load(f)
        self.dump_obs['first_0'] = torch.stack(list(self.dump_obs['first_0'])).squeeze(1).to('cpu')
        self.dump_obs['second_0'] = torch.stack(list(self.dump_obs['second_0'])).squeeze(1).to('cpu')

        # policies
        self.policy = MultiTeamPolicy(cfg).to('cpu')

        # 最新的 log prob
        self.latest_log_prob = {
            "first_0": None,
            "second_0": None
        }
        self.latest_actions = {
            "first_0": torch.stack(list(self.dump_obs['first_0_action'])).squeeze(1),
            "second_0":  torch.stack(list(self.dump_obs['second_0_action'])).squeeze(1)
        }
        self.kl = {
            "first_0": [],
            "second_0": []
        }

        self.latest_probs = {
            "first_0": [],
            "second_0":[]
        }

        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def run(self) -> None:
        if self.play_mode == "self-play":
            self.load_model_pool()

        while not self.kill_now:
            socks = dict(self._net_left.poller.poll(timeout=10))
            self.receive_model(socks, self._net_left, name="second")

            socks = dict(self._net.poller.poll(timeout=10))
            self.receive_model(socks, self._net, name="first")

            self.check_moni()

            # 发送 elo 到企业微信群
            self.check_wx()
            # 备份模型
            self.dump_model_pool()

            # TODO 计算、量化纳什均衡点
            if self._net.has_rep_data(socks):
                api, msg = self._net.receive_api_request()

                start_time = time.time()
                if api == EvaluationServer.Req.FINISH_MATCH.value:
                    self.finish_match(msg)
                elif api == EvaluationServer.Req.QUERY_TASK.value:
                    self.query_task()
                elif api == EvaluationServer.Req.GET_POOL_MODEL.value:
                    self.get_pool_model()
                elif api == EvaluationServer.Req.GET_LATEST_MODEL.value:
                    self.get_latest_model()
                elif api == EvaluationServer.Req.QUERY_EVAL.value:
                    self.query_eval()
                else:
                    self._net.send_response_api({"res": EvaluationServer.Res.INVALID_API.value})
                end_time = time.time()
                used_time = (end_time - start_time) * 1000
                # self.logger.info("handle api {0} {1} ms".format(api, used_time))

        self.logger.info("dump model pool when exit")
        self._dump_model_pool()
        self.logger.info("exit EvaluationServer gracefully")

    # 结束比赛，结算elo
    def finish_match(self, msg):
        if msg["match_type"] == "evaluate" or msg["match_type"] == "self-play" or msg["match_type"] == "self-pool":
            first_model_name = msg["first"]
            second_model_name = msg["second"]
            match_result = msg["result"]

            if first_model_name in self.elo and second_model_name in self.elo and first_model_name != second_model_name:
                first_elo_old = self.elo[first_model_name]
                second_elo_old = self.elo[second_model_name]
                first_elo, second_elo = cal_elo(first_elo_old, second_elo_old, match_result, elo_k=self.elo_k)
                self.elo[first_model_name] = first_elo
                self.elo[second_model_name] = second_elo

                eval_key = (first_model_name, second_model_name)
                if eval_key in self.history_eval_resutls:
                    self.history_eval_resutls[eval_key].append(msg['ep_reward'])
                else:
                    self.history_eval_resutls[eval_key] = [msg['ep_reward']]

                if len(self.history_eval_resutls[eval_key]) >= 50:
                    self.history_eval_resutls[eval_key] = self.history_eval_resutls[eval_key][50:]

                if first_model_name == "first_0":
                    self.first_ep_reward.append(msg['ep_reward'])
                if second_model_name == "second_0":
                    self.second_ep_reward.append(msg['ep_reward'])

                self.logger.info("first model name {}, elo: {}. second model name: {}, elo: {}.".format(first_model_name, first_elo, second_model_name, second_elo))
                self._net.send_response_api({"res": EvaluationServer.Res.OK.value})
            else:
                self._net.send_response_api({"res": EvaluationServer.Res.INVALID_PARAM.value})
        elif msg["match_type"] == "evaluate-bot":
            ep_reward = msg['ep_reward']
            self.build_in_ep_reward.append(ep_reward)
            self._net.send_response_api({"res": EvaluationServer.Res.OK.value})
        else:
            self._net.send_response_api({"res": EvaluationServer.Res.INVALID_PARAM.value})

    def get_pool_model(self):
        model_names = list(self.model_pool.keys())
        elo = np.array(list(self.elo[name] for name in model_names)) # elo 越高，采样率越高
        elo_prob = elo / elo.sum(axis=0)
        model_name = np.random.choice(model_names, p=elo_prob)

        self._net.send_response_api({
            "res": EvaluationServer.Res.OK.value,
            "model_dict": self.model_pool[model_name],
            "model_name": model_name
        })

    def get_latest_model(self):
        self._net.send_response_api({
            "res": EvaluationServer.Res.OK.value,
            "model_dict": self.latest_model_dict,
        })

    def query_task(self):
        model_time = ["first_0", "second_0"]
        model = None

        # play_mode = "build-in-bot"
        play_mode = "self-play"

        if len(self.model_pool) > 0 and np.random.rand() < 0.75:
            model_names = list(self.model_pool.keys())
            model_time = [np.random.choice(model_names)]
            model = {
                model_time[0]: self.model_pool[model_time[0]]
            }
            play_mode = "self-pool"

        self._net.send_response_api({
            "res": EvaluationServer.Res.OK.value,
            "play_mode": play_mode,
            "model_name": model_time,
            "model_config": None,
            "model_dict": model
        })

    def query_eval(self):
        # if len(self.need_to_eval_models) >= 2:
        if len(self.first_model_index) >= 2 and len(self.second_model_index) >= 2:
            while True:
                first_key = np.random.choice(list(self.first_model_index.keys()))
                second_key = np.random.choice(list(self.second_model_index.keys()))
                model_names = [first_key, second_key]
                model_time = model_names

                self.logger.info("query eval {}".format(model_names))
                if model_time[0] not in self.model_pool or model_time[1] not in self.model_pool:
                    self.logger.warn("cannot found model in model pool, {}, model key type {}".format(model_names, type(model_names[0])))
                    if len(self.first_model_index) >= 2 and len(self.second_model_index) >= 2:
                        continue
                    else:
                        break

                model = {
                    model_time[0]: self.model_pool[model_names[0]],
                    model_time[1]: self.model_pool[model_names[1]]
                }
                play_mode = 'evaluate'

                self._net.send_response_api({
                    "res": EvaluationServer.Res.OK.value,
                    "play_mode": play_mode,
                    "model_name": model_time,
                    "model_config": None,
                    "model_dict": model
                })

                break
        else:
            self._net.send_response_api({
                "res": EvaluationServer.Res.NOT_FOUND_MODEL.value,
            })

    def nash_conv(self):
        model_len = len(self.model_pool)
        first_model_len = len(self.first_model_index)
        second_model_len = len(self.second_model_index)
        self.logger.info(list(self.model_pool.keys()))
        self.logger.info(self.history_eval_resutls)

        if model_len < 2 or first_model_len < 2 or second_model_len < 2:
            return 0, 0, 0, 0

        M = np.zeros((first_model_len, second_model_len))
        try:
            first_max_M = np.zeros((first_model_len))
            second_max_M = np.zeros((second_model_len))

            for keys, value in self.history_eval_resutls.items():
                first_key, second_key = keys
                first_ind = self.first_model_index[first_key]
                second_ind = self.second_model_index[second_key]
                M[first_ind][second_ind] = np.mean(value)

            first_max_M = np.max(M, axis=0)
            # nan -> 0
            first_max_M[np.isnan(first_max_M)] = 0
            first_mean_M = np.mean(M, axis=0)
            first_mean_M[np.isnan(first_mean_M)] = 0
            # second_max_M = np.max(-M, axis=1)

            self.logger.info("M: {}".format(M))

            # 计算 M_updated
            # # 不考虑对角线的归一化和熵计算
            M_normalized = M / M.sum(axis=1, keepdims=True)

            # 计算熵
            H = -np.sum(M_normalized * np.log(M_normalized+1e-9)) # 加上小量避免log(0)

            # 计算非对角线元素的方差和标准差
            variance = np.var(M[M != 0])
            std_dev = np.sqrt(variance)

            # 用最大reward - 当前reward 作为 nash conv
            nash_conv_value = np.mean(first_max_M - first_mean_M)
            self.logger.info("nash conv: {}".format(nash_conv_value))

            return nash_conv_value, H, variance, std_dev
        except Exception as e:
            self.logger.error("nash conv error: {} , \n {}, \n {}".format(repr(e), self.second_model_index.keys(), M))
            return 0, 0, 0, 0

    def receive_model(self, socks, net, name="first"):
        model_str = None
        if net.subscriber in socks and socks[net.subscriber] == zmq.POLLIN:
            model_str = net.subscriber.recv()

        if model_str is None:
            return

        model_dict = pickle.loads(model_str)

        # to cpu
        new_dict = {}
        for k, v in model_dict.items():
            if k.startswith("module"):
                new_k = k[7:]
            else:
                new_k = k

            if type(v) is dict:
                new_dict[new_k] = {k: v.cpu() for k, v in v.items()}
            else:
                new_dict[new_k] = v.cpu()

        del model_dict
        model_dict = new_dict

        self.latest_model_dict = model_dict # CPU
        # self.logger.info("evaluation receive model: {}".format(model_dict["_update_time"]))
        self.policy.load_state_dict(new_dict) # cpu
        self.policy.to('cpu')
        # 计算 logits

        flag = False
        kl = 0
        with torch.no_grad():
            if name == "first":
                actions, log_probs, value, probs = self.policy(self.dump_obs['first_0']) # 128 samples
                if self.latest_log_prob['first_0'] is not None:
                    new_log_prob = probs.log_prob(self.latest_actions['first_0'])
                    # kl = (new_log_prob - self.latest_log_prob['first_0']).mean().detach().numpy()
                    # self.kl['first_0'].append(np.clip(kl, -10.0, 10))

                    kl = []
                    for prob in self.latest_probs['first_0'][-50:]:
                        lakl = (new_log_prob - prob.log_prob(self.latest_actions['first_0'])).mean().detach().numpy()
                        kl.append(lakl)

                    kl = np.mean(kl)
                    self.kl['first_0'].append(np.clip(kl, -10.0, 10))

                    flag = np.abs(kl) > 0.35
                    # show devices
                    self.logger.info("evaluation receive model: {} {} kl: {}".format(name, model_dict["_update_time"], kl))
                else:
                    flag = True
            else:
                actions, log_probs, value, probs = self.policy(self.dump_obs['second_0'])
                if self.latest_log_prob['second_0'] is not None:
                    new_log_prob = probs.log_prob(self.latest_actions['second_0'])
                    # kl = (new_log_prob - self.latest_log_prob['second_0']).mean().detach().numpy()

                    kl = []
                    for prob in self.latest_probs['second_0'][-50:]:
                        lakl = (new_log_prob - prob.log_prob(self.latest_actions['second_0'])).mean().detach().numpy()
                        kl.append(lakl)

                    kl = np.mean(kl)

                    self.kl['second_0'].append(np.clip(kl, -10.0, 10))
                    flag = np.abs(kl) > 0.35
                    self.logger.info("evaluation receive model: {} {} kl: {}".format(name, model_dict["_update_time"], kl))
                else:
                    flag = True

        if "_update_time" in model_dict and self._should_update(model_time=float(model_dict['_update_time'].cpu().detach().numpy())) and flag:
            model_key = name + "_" + str(model_dict['_update_time'].cpu().detach().numpy())
            self._insert_new_model(model_key=model_key, model_dict=model_dict)
            if flag:
                if name == "first":
                    self.latest_log_prob['first_0'] = log_probs
                    self.latest_actions['first_0'] = actions.squeeze(1)
                    self.latest_probs['first_0'].append(probs)
                else:
                    self.latest_log_prob['second_0'] = log_probs
                    self.latest_actions['second_0'] = actions.squeeze(1)
                    self.latest_probs['second_0'].append(probs)

    def _should_update(self, model_time):
        return model_time not in self.model_pool and time.time() > self.update_model_time

    def _insert_new_model(self, model_key, model_dict, elo=None):
        self.model_pool[model_key] = model_dict
        if elo is not None:
            self.elo[model_key] = elo
        else:
            self.elo[model_key] = 1500.0

        if 'first' in model_key:
            self.first_model_index[model_key] = len(self.first_model_index)
        else:
            self.second_model_index[model_key] = len(self.second_model_index)

        self.logger.info("add new model {}".format(model_key))
        self.logger.info("model pool size: {}".format(len(self.model_pool)))

        if is_development():
            self.update_model_time = time.time() + 30
        else:
            # delta = min((time.time() - self.start_time + 600) / 3600, 1)
            self.update_model_time = time.time() + 60 * 1

        # remove minimum elo model
        # if len(self.model_pool) > self.population_n:
        #     min_elo = 10000
        #     min_elo_model_time = None
        #     for t, _ in self.model_pool.items():
        #         if t in self.elo and self.elo[t] < min_elo:
        #             min_elo = self.elo[t]
        #             min_elo_model_time = t

        #     if min_elo_model_time == 1500.0:
        #         # find the first model
        #         min_elo_model_time = list(self.model_pool.keys())[0]

        #     if min_elo_model_time is not None:
        #         self.model_pool.pop(min_elo_model_time)
        #         self.elo.pop(min_elo_model_time)
        #         for models_list in self.need_to_eval_models:
        #             if min_elo_model_time in models_list:
        #                 self.need_to_eval_models.remove(models_list)

        #         self.logger.info("pop model {}".format(min_elo_model_time))


        # add new model to need to eval list
        if len(self.model_pool) >= 2 and elo is None:
            for t, _ in self.model_pool.items():
                if model_key != t:
                    self.need_to_eval_models.append([t, model_key])

            self.logger.info("add new model to need to eval list {}".format(self.need_to_eval_models))

        self.logger.info("model pool size: {}, model keys: {}".format(len(self.model_pool), list(self.model_pool.keys())))

    def check_moni(self):
        if time.time() > self.moni_check_time:
            moni_data = {
                "model_pool_length": len(self.model_pool),
                "eval_size": len(self.history_eval_resutls),
                "first_elo": self.elo['first_0'],
                "second_elo": self.elo['second_0'],
            }

            nash_conv, H_no_diag, variance, std_dev = self.nash_conv()
            if nash_conv != 0:
                moni_data['nash_conv'] = nash_conv
            if H_no_diag != 0 and variance != 0 and std_dev != 0:
                moni_data['H_no_diag'] = H_no_diag
                moni_data['variance'] = variance
                moni_data['std_dev'] = std_dev

            if len(self.build_in_ep_reward) > 10:
                moni_data["build_in_ep_reward"] = np.mean(self.build_in_ep_reward)
                self.build_in_ep_reward = []

            if len(self.first_ep_reward) > 10:
                moni_data["first_ep_reward"] = np.mean(self.first_ep_reward)
                self.first_ep_reward = []

            if len(self.second_ep_reward) > 10:
                moni_data["second_ep_reward"] = np.mean(self.second_ep_reward)
                self.second_ep_reward = []

            if len(self.kl['first_0']) > 5:
                moni_data['first_0_kl'] = np.mean(self.kl['first_0'])
                self.kl['first_0'] = []

            if len(self.kl['second_0']) > 5:
                moni_data['second_0_kl'] = np.mean(self.kl['second_0'])
                self.kl['second_0'] = []


            # all_count = len(self.model_result_table["training"])
            # if all_count != 0:
            #     moni_data.update({
            #         "win_prob_training": self.model_result_table["training"].count("win") / all_count,
            #         "draw_prob_training": self.model_result_table["training"].count("draw") / all_count,
            #     })
            self.ls_api.record(msg_type="evaluation_server", data=moni_data)
            self.ls_api.send_moni(msg_type="evaluation_server")
            self.moni_check_time = time.time() + 30

    def check_wx(self):
        pass

    # 加载本地模型
    def load_model_pool(self):
        if os.path.exists("./log/model_pools") is False:
            return
        files = os.listdir("./log/model_pools")
        for file in files:
            if "model_f" in file:
                with open(f"./log/model_pools/{file}", "rb") as f:
                    model_dict = pickle.loads(f.read())
                    elo = float(file.split("_")[-1])
                    if 'first' in file:
                        model_key = 'first' + '_' + str(model_dict['_update_time'].cpu().detach().numpy())
                        self._insert_new_model(model_key=model_key, model_dict=model_dict, elo=elo)
                    else:
                        model_key = 'second' + '_' + str(model_dict['_update_time'].cpu().detach().numpy())
                        self._insert_new_model(model_key=model_key, model_dict=model_dict, elo=elo)

        self.logger.info("load model pool successfully!")

    def dump_model_pool(self):
        if time.time() >= self.next_dump_time:
            self.next_dump_time = time.time() + 60 if is_development() else time.time() + 60 * 60
            self.logger.info("training elo: {0}".format(self.elo["training"]))
            self.logger.info("pool elo: {0}".format(self.elo))
            self._dump_model_pool()

    # 备份模型
    def _dump_model_pool(self):
        self.logger.info(f"dump model pool {len(self.model_pool)}")
        model_path = "./log/model_pools" # if is_development() else "/model_dir"

        create_path(model_path)
        for model_time, model_dict in self.model_pool.items():
            elo = 0
            if model_time in self.elo:
                elo = self.elo[model_time]

            file = f"model_f_{model_time}_{elo}"
            with open(f"{model_path}/{file}", "wb") as f:
                f.write(pickle.dumps(model_dict))

    def exit_gracefully(self, signum, frame):
        self.kill_now = True
        self.logger.info(f"get exit_gracefully signal {signum} {frame}")

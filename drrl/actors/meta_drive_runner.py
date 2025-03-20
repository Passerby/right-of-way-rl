import time
import random
import concurrent.futures

import gymnasium as gym
import numpy as np

from drrl.envs.meta_drive import meta_drive_env, make_ai_vs_bot_env
from drrl.common.utils import setup_logger
from drrl.actors.multi_agent_ppo_team import MultiAgentPPOTeam


class MetaDriveRunner:

    def __init__(self, cfg, play_mode='self-play') -> None:
        self.logger = setup_logger("worker_runner.log")
        self.logger.info(f"start runner {play_mode} mode")
        # self.es_api = EvaluationClient(cfg)

        self.cfg = cfg
        self.seed = None
        self.model_names = []
        self.game_env = None # type: gym.Env
        # self-play FSP 训练
        # evaluate 本地评估两个模型
        # evaluate-bot 评估训练模型和内置bot
        self.play_mode = play_mode # 'build-in-bot' # query from evaluation server
        self.match_results = []

        self.start_state = True
        self.run_times = 0

        self.teams = None
        self.action_space = None
        self.max_players = cfg['policy']['placeholder']['max_agents']
        self.agent_index_to_name = {}

    def run(self):
        n_timestep = 0
        raw_rewards = None
        dones = None

        while True:
            if self.start_state:
                n_timestep = 0
                observations, infos = self._before_match()

                if raw_rewards is None:
                    raw_rewards = {}
                    dones = {}
                    for idx, agent_name in enumerate(self.game_env.agents.keys()):
                        if self.play_mode == "build-in-bot" or self.play_mode == "evaluate-bot":
                            raw_rewards = 0
                        else:
                            raw_rewards[agent_name] = 0
                        dones[agent_name] = False
                        self.agent_index_to_name[agent_name] = idx
                else:
                    for agent_name in self.game_env.agents.keys():
                        if self.play_mode == "build-in-bot" or self.play_mode == "evaluate-bot":
                            raw_rewards = 0
                        else:
                            if agent_name not in raw_rewards:
                                raw_rewards[agent_name] = 0

                        if agent_name not in dones:
                            dones[agent_name] = False

                if self.action_space is None:
                    if self.play_mode == "build-in-bot":
                        self.action_space = self.game_env.action_space
                    else:
                        default_first_agent_key = list(self.game_env.action_space.keys())[0]
                        self.action_space = self.game_env.action_space[default_first_agent_key]

            all_actions = None

            start_time = time.time()
            threads = []

            if self.play_mode == "build-in-bot" or self.play_mode == "evaluate-bot":
                all_actions = []
                action = self.teams.act(observations, observations, last_reward=raw_rewards, last_done=dones['default_agent'], info=infos)
                all_actions = action
                # print("actions: ", all_actions)
            else:
                all_actions = {}
                all_observations = np.zeros((self.max_players * 91))
                for agent_name, o in observations.items():
                    obs_index = self.agent_index_to_name[agent_name]
                    all_observations[obs_index * 91:(obs_index + 1) * 91] = o
                all_actions = self.teams.act(observations, all_observations, last_reward=raw_rewards, last_done=dones, info=infos)

            end_time = time.time()

            used_time = (end_time - start_time) * 1000
            # self.logger.info(f"inference used_time: {used_time} ms")

            # observations, rewards, terminations, truncations, infos
            if self.play_mode == "build-in-bot":
                observations, raw_rewards, terminations, truncations, infos = self.game_env.step(all_actions['agent0'])
                dones['defualt_agent'] = terminations or truncations
            else:
                observations, raw_rewards, terminations, truncations, infos = self.game_env.step(all_actions)
                for agent_id, v in terminations.items():
                    if v or truncations[agent_id]:
                        dones[agent_id] = True

            step_time = (time.time() - end_time) * 1000

            # MA 任意 crash 时, 全部都被终止时
            final_done = np.logical_or(terminations, truncations) if self.play_mode in ['build-in-bot', 'evaluate-bot'] else terminations['__all__'] or truncations['__all__'] # np.logical_or.reduce(list(terminations.values()) + list(truncations.values()))

            if final_done: # or n_timestep >= 2048 * 2:
                self._finish_match()
            n_timestep += 1

    def _before_match(self):
        self.start_state = False
        self.logger.info("before match start")

        model_dicts = {}
        # env init
        if self.run_times == 0:
            if self.play_mode == 'build-in-bot' or self.play_mode == 'evaluate-bot':
                self.game_env = make_ai_vs_bot_env(self.cfg)
            else:
                self.game_env = meta_drive_env(self.cfg)

        observations, infos = self.game_env.reset(seed=self.seed)

        # team init
        if self.run_times % 1000 == 0:
            del self.teams
            self.teams = {}

            team = MultiAgentPPOTeam(
                team_id="default",
                cfg=self.cfg,
                policy=None,
                inference=True,
                local_test=False,
            )


            self.teams = team


        return observations, infos

    def _finish_match(self):
        self.start_state = True

        end_info = {}

        self.teams.send_end_info(end_info)
        self.teams.state_reset()

        # for team_id, team_agent in self.teams.items():
        #     if type(team_agent) == MultiAgentPPOTeam:
        #         self.logger.info(f"send end info ep_step {team_agent.ep_step}")
        #         team_agent.send_end_info(end_info)
        #         team_agent.state_reset()

        self.run_times += 1


def start_thread_act(team_idx, team, observation, all_observations, rewards, infos, dones, action_space):
    if type(team) == MultiAgentPPOTeam:
        action = team.act(observation, all_observations, last_reward=rewards, done=dones, info=infos)
    else:
        action = action_space.sample()

    return action, team_idx

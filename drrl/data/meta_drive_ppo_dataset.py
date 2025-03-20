import collections
import random

import numpy as np
import torch

from drrl.data.ppo_data_frame import PPODataFrame as DataFrame


class MetaDrivePPODataSet:

    def __init__(self, max_capacity: int, data_type="training_server", device="cpu"):
        self.max_capacity = max_capacity
        self.data_type = data_type
        self.device = device

        self.data_frames = {
            "actor_obs": collections.deque(maxlen=self.max_capacity),
            "critic_obs": collections.deque(maxlen=self.max_capacity),
            "next_obs": collections.deque(maxlen=self.max_capacity),
            "lstm_state": collections.deque(maxlen=self.max_capacity),
            "mask": collections.deque(maxlen=self.max_capacity),
            "done": collections.deque(maxlen=self.max_capacity),
            # 仅在 training_server 模式下添加
            "action": collections.deque(maxlen=self.max_capacity),
            "adv": collections.deque(maxlen=self.max_capacity),
            "returns": collections.deque(maxlen=self.max_capacity),
            "reward": collections.deque(maxlen=self.max_capacity),
            "value": collections.deque(maxlen=self.max_capacity),
            "logprob": collections.deque(maxlen=self.max_capacity),
            "model_update_times": collections.deque(maxlen=self.max_capacity)
        }

        self.inference_keys = [
            "actor_obs",
            "critic_obs",
            "lstm_state",
            "mask",
            "done",
        ]

    def __len__(self):
        return len(self.data_frames["actor_obs"])

    @property
    def cur_size(self):
        return len(self.data_frames["actor_obs"])

    def append(self, instance: DataFrame):
        # self.data_frames.append(instance)
        if self.data_type == 'training_server':
            for key in self.data_frames.keys():
                self.data_frames[key].append(getattr(instance, key))
        else:
            for key in self.inference_keys:
                self.data_frames[key].append(getattr(instance, key))

    def prepare_for_inference(self):
        # 将所有的 observation 转换为 numpy 数组
        batch_size = len(self.data_frames["actor_obs"])

        batch_dict = {}
        batch_dict["actor_obs"] = torch.tensor(
            np.array(self.data_frames["actor_obs"]), dtype=torch.float32).reshape(batch_size, -1).to(self.device)

        batch_dict["critic_obs"] = torch.tensor(
            np.array(self.data_frames["critic_obs"]), dtype=torch.float32).reshape(batch_size, -1).to(self.device)

        batch_dict['done'] = torch.tensor(
            np.array(self.data_frames["done"]), dtype=torch.float32).reshape(batch_size, -1).to(self.device)

        return batch_dict

    def prepare_for_training(self):
        batch_size = len(self.data_frames["actor_obs"])
        batch_dict = {}

        # basic information
        batch_dict["actor_obs"] = torch.tensor(
            np.array(self.data_frames["actor_obs"]), dtype=torch.float32).reshape(batch_size, -1).to(self.device)
        batch_dict["critic_obs"] = torch.tensor(
            np.array(self.data_frames["critic_obs"]), dtype=torch.float32).reshape(batch_size, -1).to(self.device)
        batch_dict["next_obs"] = torch.tensor(
            np.array(self.data_frames["next_obs"]), dtype=torch.float32).reshape(batch_size, -1).to(self.device)
        batch_dict["action"] = torch.tensor(
            np.array(self.data_frames["action"]), dtype=torch.float32).reshape(batch_size, -1).to(self.device)
        batch_dict["adv"] = torch.tensor(
            np.array(self.data_frames["adv"]), dtype=torch.float32).reshape(batch_size, -1).to(self.device)
        batch_dict["returns"] = torch.tensor(
            np.array(self.data_frames["returns"]), dtype=torch.float32).reshape(batch_size, -1).to(self.device)
        batch_dict["reward"] = torch.tensor(
            np.array(self.data_frames["reward"]), dtype=torch.float32).reshape(batch_size, -1).to(self.device)
        batch_dict["value"] = torch.tensor(
            np.array(self.data_frames["value"]), dtype=torch.float32).reshape(batch_size, -1).to(self.device)
        batch_dict["logprob"] = torch.tensor(
            np.array(self.data_frames["logprob"]), dtype=torch.float32).reshape(batch_size, -1).to(self.device)
        batch_dict["done"] = torch.tensor(
            np.array(self.data_frames["done"]), dtype=torch.float32).reshape(batch_size, -1).to(self.device)
        batch_dict["model_update_times"] = torch.tensor(
            np.array(self.data_frames["model_update_times"]), dtype=torch.float32).reshape(batch_size, -1).to(self.device)

        return batch_dict

    def sample(self, batch_size):
        max_len = len(self.data_frames["actor_obs"])
        if max_len < batch_size:
            return None

        # sample_index = random.sample(range(max_len), batch_size)
        batch_dict = {}

        batch_dict['actor_obs'] = np.array(self.data_frames['actor_obs'])
        batch_dict['critic_obs'] = np.array(self.data_frames['critic_obs'])
        batch_dict['next_obs'] = np.array(self.data_frames['next_obs'])
        batch_dict['action'] = np.array(self.data_frames['action'])
        batch_dict['adv'] = np.array(self.data_frames['adv'])
        batch_dict['returns'] = np.array(self.data_frames['returns'])
        batch_dict['reward'] = np.array(self.data_frames['reward'])
        batch_dict['value'] = np.array(self.data_frames['value'])
        batch_dict['logprob'] = np.array(self.data_frames['logprob'])
        batch_dict['done'] = np.array(self.data_frames['done'])
        batch_dict['model_update_times'] = np.array(self.data_frames['model_update_times'])

        return batch_dict

    def clear(self):
        self.data_frames = {
            "actor_obs": collections.deque(maxlen=self.max_capacity),
            "critic_obs": collections.deque(maxlen=self.max_capacity),
            "lstm_state": collections.deque(maxlen=self.max_capacity),
            "mask": collections.deque(maxlen=self.max_capacity),
            "done": collections.deque(maxlen=self.max_capacity),
            # 仅在 training_server 模式下添加
            "action": collections.deque(maxlen=self.max_capacity),
            "adv": collections.deque(maxlen=self.max_capacity),
            "returns": collections.deque(maxlen=self.max_capacity),
            "reward": collections.deque(maxlen=self.max_capacity),
            "value": collections.deque(maxlen=self.max_capacity),
            "logprob": collections.deque(maxlen=self.max_capacity),
            "model_update_times": collections.deque(maxlen=self.max_capacity)
        }

    def fit_max_size(self):
        if self.cur_size > self.max_capacity:
            keep_index_start = self.cur_size - self.max_capacity
            # obs_keep_index_start = keep_index_start * self.player_num

            for key in ['actor_obs', 'critic_obs', 'lstm_state', 'mask', 'done']:
                self.data_frames[key] = self.data_frames[key][keep_index_start:]

            if self.data_type == 'training_server':
                for key in ['next_obs', 'action', 'adv', 'returns', 'reward', 'value', 'logprob', 'model_update_times']:
                    self.data_frames[key] = self.data_frames[key][keep_index_start:]

    def get_average_update_time(self):
        update_times = [df for df in self.data_frames['model_update_times'] if df is not None]
        return np.mean(update_times) if update_times else 0

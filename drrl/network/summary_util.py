import time
from enum import Enum

import numpy as np
from torch.utils.tensorboard import SummaryWriter


class MessageType(Enum):
    """ 枚举数据传输类型
    name_server
    agent   （actor的信息）
    inference_server
    training_server
    model   （模型训练返回的信息）
    evaluation_server
    """
    AGENT = "agent"
    MODEL = "model"
    INFERENCE_SERVER = "inference_server"
    NAME_SERVER = "name_server"
    EVALUATION_SERVER = "evaluation_server"
    TRAINING_SERVER = "training_server"


class SummaryLog:

    def __init__(self, summary_writer: SummaryWriter):
        self.summary_writer = summary_writer
        self.tag_values_dict = {}
        self.tag_step_dict = {}
        self.tag_output_freq_dict = {}
        self.tag_func_dict = {}
        self.tag_total_add_count = {}
        self.total_env_steps = 0
        self.tag_last_time_dict = {}

    def add_tag(self, tag, output_freq, tag_func):
        self.tag_values_dict[tag] = []
        self.tag_step_dict[tag] = 0
        self.tag_output_freq_dict[tag] = output_freq
        self.tag_func_dict[tag] = tag_func
        self.tag_total_add_count[tag] = 0
        if self.tag_func_dict[tag] in ["per_min", "per_second"]:
            self.tag_last_time_dict[tag] = time.time()

    def has_tag(self, tag):
        if tag in self.tag_step_dict.keys():
            return True
        else:
            return False

    def get_tag_count(self, tag):
        return self.tag_total_add_count[tag]

    def list_add_summary(self, tag, values):
        if tag not in self.tag_values_dict:
            return

        self.tag_total_add_count[tag] += 1

        result_value = 0

        self.tag_values_dict[tag] += values

        if len(self.tag_values_dict[tag]) >= self.tag_output_freq_dict[tag]:
            if self.tag_func_dict[tag] == "avg":
                result_value = sum(self.tag_values_dict[tag]) / len(self.tag_values_dict[tag])
            elif self.tag_func_dict[tag] == "total":
                result_value = sum(self.tag_values_dict[tag])
            elif self.tag_func_dict[tag] == "max":
                result_value = max(self.tag_values_dict[tag])
            elif self.tag_func_dict[tag] == "min":
                result_value = min(self.tag_values_dict[tag])
            elif self.tag_func_dict[tag] == "std":
                result_value = np.array(self.tag_values_dict[tag]).std()
            elif self.tag_func_dict[tag] == "per_min":
                result_value = sum(self.tag_values_dict[tag]) * 60 / (time.time() - self.tag_last_time_dict[tag])
            elif self.tag_func_dict[tag] == "per_second":
                result_value = sum(self.tag_values_dict[tag]) / (time.time() - self.tag_last_time_dict[tag])

            if "agent" in tag:
                summary_steps = self.total_env_steps
            else:
                summary_steps = self.tag_step_dict[tag]

            self.summary_writer.add_scalar(tag, result_value, summary_steps)

            self.tag_step_dict[tag] += 1
            self.tag_values_dict[tag] = []
            if self.tag_func_dict[tag] in ["per_min", "per_second"]:
                self.tag_last_time_dict[tag] = time.time()

    def add_summary(self, tag, value):
        if tag not in self.tag_values_dict:
            return

        self.tag_total_add_count[tag] += 1

        if "agent" in tag:
            summary_steps = self.total_env_steps
        else:
            summary_steps = self.tag_step_dict[tag]

        self.summary_writer.add_scalar(tag, value, summary_steps)

        self.tag_step_dict[tag] += 1

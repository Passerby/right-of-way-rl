import datetime
import os
import pickle
import time
import json

import zmq
from torch.utils.tensorboard import SummaryWriter

from drrl.common.utils import setup_logger
from drrl.network.name_client import NameClient
from drrl.network.zmq import ZmqAdaptor
from drrl.network.summary_util import SummaryLog, MessageType


class LogServer:
    """
    监控指标分五部分：

    name_server
    agent   (actor的信息)
    inference_server
    training_server
    model   (模型训练返回的信息)
    evaluation_server

    根据 log server 接收数据中的 msg_type 信息来分别写入 tensorboard 中。
    """

    def __init__(self, cfg):
        if not os.path.exists("./summary_log/"):
            os.makedirs("./summary_log/", exist_ok=True)
        self.cfg = cfg
        self.logger = setup_logger("./log_server.log")
        self.zmq_adaptor = ZmqAdaptor(logger=self.logger)

        ns_api = NameClient(cfg)
        _address, port = ns_api.register(rtype="log_server", extra={"data_type": "log_server"})
        self.zmq_adaptor.start({"mode": "pull", "host": "*", "port": port})

        self.raw_data_list = []
        self.next_print_time = 0

        exp_name = self.cfg.log_server["exp_name"]
        time_now = datetime.datetime.now().strftime("%m%d%H%M%S")
        self.summary_writer = SummaryWriter(f"./summary_log/log_{cfg.env.env_name}_agents_{cfg.policy.placeholder.max_agents}_{cfg.env.phi}_{cfg.env.neighbours_distance}_{cfg.env.safe_distance_ratio}_{time_now}_{exp_name}")
        self.summary_logger = SummaryLog(self.summary_writer)

        self.worker_uid_dict = {}
        self.inf_server_uid_dict = {}
        self.next_check_distinct_worker_uid_time = time.time() + 60 * 3
        self.next_check_distinct_inf_server_uid_time = time.time() + 60 * 3

        # tags
        self.name_server_tags_config = self.cfg.moni[MessageType.NAME_SERVER.value]
        self.name_server_tags = []
        self.agent_tags_config = self.cfg.moni[MessageType.AGENT.value]
        self.agent_tags = []
        self.inf_server_tags_config = self.cfg.moni[MessageType.INFERENCE_SERVER.value]
        self.inf_server_tags = []
        self.train_tags_config = self.cfg.moni[MessageType.TRAINING_SERVER.value]
        self.train_tags = []
        self.model_tags_config = self.cfg.moni[MessageType.MODEL.value]
        self.model_tags = []
        self.evaluate_server_tags_config = self.cfg.moni[MessageType.EVALUATION_SERVER.value]
        self.evaluate_server_tags = []

    def summary_definition(self):
        # name_server
        for tag_func, tag_dict in self.name_server_tags_config.items():
            for tag_item in tag_dict:
                tag, output_freq = tag_item["tag"], tag_item["output_freq"]
                self.name_server_tags.append(tag)
                self.summary_logger.add_tag("{0}/{1}_{2}".format(MessageType.NAME_SERVER.value, tag, tag_func), output_freq,
                                            tag_func)

        # agent
        for tag_func, tag_dict in self.agent_tags_config.items():
            for tag_item in tag_dict:
                tag, output_freq = tag_item["tag"], tag_item["output_freq"]
                self.agent_tags.append(tag)
                self.summary_logger.add_tag("{0}/{1}_{2}".format(MessageType.AGENT.value, tag, tag_func), output_freq, tag_func)

        # inference_server
        for tag_func, tag_dict in self.inf_server_tags_config.items():
            for tag_item in tag_dict:
                tag, output_freq = tag_item["tag"], tag_item["output_freq"]
                self.inf_server_tags.append(tag)
                self.summary_logger.add_tag("{0}/{1}_{2}".format(MessageType.INFERENCE_SERVER.value, tag, tag_func),
                                            output_freq, tag_func)

        # training_server
        for tag_func, tag_dict in self.train_tags_config.items():
            for tag_item in tag_dict:
                tag, output_freq = tag_item["tag"], tag_item["output_freq"]
                self.train_tags.append(tag)
                self.summary_logger.add_tag("{0}/{1}_{2}".format(MessageType.TRAINING_SERVER.value, tag, tag_func), output_freq,
                                            tag_func)

        # model
        for tag_func, tag_dict in self.model_tags_config.items():
            for tag_item in tag_dict:
                tag, output_freq = tag_item["tag"], tag_item["output_freq"]
                self.model_tags.append(tag)
                self.summary_logger.add_tag("{0}/{1}_{2}".format(MessageType.MODEL.value, tag, tag_func), output_freq, tag_func)

        # evaluation_server
        for tag_func, tag_dict in self.evaluate_server_tags_config.items():
            for tag_item in tag_dict:
                tag, output_freq = tag_item["tag"], tag_item["output_freq"]
                self.evaluate_server_tags.append(tag)
                self.summary_logger.add_tag("{0}/{1}_{2}".format(MessageType.EVALUATION_SERVER.value, tag, tag_func),
                                            output_freq, tag_func)

        # text summary
        # dump cfg to text summary
        t = str(self.cfg)
        self.summary_writer.add_text("cfg", t)

        self.logger.info("name_server_tags, {}".format(self.name_server_tags))
        self.logger.info("agent_tags, {}".format(self.agent_tags))
        self.logger.info("inf_server_tags, {}".format(self.inf_server_tags))
        self.logger.info("trains_tags, {}".format(self.train_tags))
        self.logger.info("model_tags, {}".format(self.model_tags))
        self.logger.info("evaluate_server_tags, {}".format(self.evaluate_server_tags))

    def log_detail(self, data):
        if data.get("msg_type") == MessageType.NAME_SERVER.value:
            for tag_func, tag_dict in self.name_server_tags_config.items():
                for data_key in data.keys():
                    if data_key in self.name_server_tags:
                        tag = data_key
                        if type(data[tag]) is list:
                            values = data[tag]
                        else:
                            values = [data[tag]]
                        self.summary_logger.list_add_summary("{0}/{1}_{2}".format(MessageType.NAME_SERVER.value, tag, tag_func),
                                                             values)

        if data.get("msg_type") == MessageType.AGENT.value:
            self.worker_uid_dict[data["uid"]] = 0
            if "env_steps" in data.keys():
                self.summary_logger.total_env_steps += sum(data["env_steps"])
            self.summary_logger.add_summary("{0}/env_steps_total".format(MessageType.AGENT.value),
                                            self.summary_logger.total_env_steps)

            # if "actions" in data.keys():
            #     for i in range(self.cfg.train["action_shape"]):
            #         self.summary_logger.list_add_summary("{0}/action_{1}_avg".format(MessageType.AGENT.value, i),
            #                                              [data["actions"][0][i]])

            # check if all docker alive every 5 min
            if time.time() > self.next_check_distinct_worker_uid_time:
                self.next_check_distinct_worker_uid_time = time.time() + 60 * 5
                self.summary_logger.add_summary("{0}/docker_num_5m_avg".format(MessageType.AGENT.value),
                                                len(self.worker_uid_dict))
                self.worker_uid_dict = {}

            for tag_func, tag_dict in self.agent_tags_config.items():
                for data_key in data.keys():
                    if data_key in self.agent_tags:
                        tag = data_key
                        if type(data[tag]) is list:
                            values = data[tag]
                        else:
                            values = [data[tag]]
                        self.summary_logger.list_add_summary("{0}/{1}_{2}".format(MessageType.AGENT.value, tag, tag_func),
                                                             values)

        if data.get("msg_type") == MessageType.INFERENCE_SERVER.value:
            self.inf_server_uid_dict[data["uid"]] = 0
            if time.time() > self.next_check_distinct_inf_server_uid_time:
                self.next_check_distinct_inf_server_uid_time = time.time() + 60 * 5
                self.summary_logger.add_summary("{0}/inf_server_count_avg".format(MessageType.INFERENCE_SERVER.value),
                                                len(self.inf_server_uid_dict))
                self.inf_server_uid_dict = {}
            for tag_func, tag_dict in self.inf_server_tags_config.items():
                for data_key in data.keys():
                    if data_key in self.inf_server_tags:
                        tag = data_key
                        if type(data[tag]) is list:
                            values = data[tag]
                        else:
                            values = [data[tag]]
                        self.summary_logger.list_add_summary(
                            "{0}/{1}_{2}".format(MessageType.INFERENCE_SERVER.value, tag, tag_func), values)

        if data.get("msg_type") == MessageType.TRAINING_SERVER.value:
            if "training_steps_total" in data.keys():
                self.summary_logger.add_summary("{0}/training_steps_total".format(MessageType.TRAINING_SERVER.value),
                                                max(data.pop("training_steps_total")))
            for tag_func, tag_dict in self.train_tags_config.items():
                for data_key in data.keys():
                    if data_key in self.train_tags:
                        tag = data_key
                        if type(data[tag]) is list:
                            values = data[tag]
                        else:
                            values = [data[tag]]
                        self.summary_logger.list_add_summary(
                            "{0}/{1}_{2}".format(MessageType.TRAINING_SERVER.value, tag, tag_func), values)

        if data.get("msg_type") == MessageType.MODEL.value:
            for tag_func, tag_dict in self.model_tags_config.items():
                for data_key in data.keys():
                    if data_key in self.model_tags:
                        tag = data_key
                        if type(data[tag]) is list:
                            values = data[tag]
                        else:
                            values = [data[tag]]
                        self.summary_logger.list_add_summary("{0}/{1}_{2}".format(MessageType.MODEL.value, tag, tag_func),
                                                             values)

        if data.get("msg_type") == MessageType.EVALUATION_SERVER.value:
            for tag_func, tag_dict in self.evaluate_server_tags_config.items():
                for data_key in data.keys():
                    if data_key in self.evaluate_server_tags:
                        tag = data_key
                        if type(data[tag]) is list:
                            values = data[tag]
                        else:
                            values = [data[tag]]
                        self.summary_logger.list_add_summary(
                            "{0}/{1}_{2}".format(MessageType.EVALUATION_SERVER.value, tag, tag_func), values)

    def run(self):
        self.summary_definition()
        self.logger.info('log begin')
        while True:
            if time.time() > self.next_print_time:
                self.summary_writer.flush()
                self.next_print_time = time.time() + 10
            socks = dict(self.zmq_adaptor.poller.poll(timeout=100))

            if self.zmq_adaptor.receiver in socks and socks[self.zmq_adaptor.receiver] == zmq.POLLIN:
                while True:
                    try:
                        data = self.zmq_adaptor.receiver.recv(zmq.NOBLOCK)
                        self.raw_data_list.append(data)
                    except zmq.ZMQError as e:
                        if type(e) != zmq.error.Again:
                            self.logger.warn("recv zmq {}".format(e))
                        break
            for raw_data in self.raw_data_list:
                data = pickle.loads(raw_data)
                if type(data) == dict:
                    data = [data]
                for log in data:
                    if "msg_type" in log and log["msg_type"] == "error":
                        if "error_msg" in log:
                            self.logger.error(log["error"])
                    self.log_detail(log)

            self.raw_data_list = []
            time.sleep(1)

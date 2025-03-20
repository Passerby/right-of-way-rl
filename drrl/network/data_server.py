import pickle
import time

import zmq

from drrl.common.utils import setup_logger
from drrl.network.name_client import NameClient
from drrl.network.zmq import ZmqAdaptor
from drrl.data.meta_drive_ppo_dataset import MetaDrivePPODataSet as PPODataSet


class DataServer:

    def __init__(self, cfg, index, device, team_id=None):
        self.cfg = cfg
        self.index = index
        self.logger = setup_logger("data_server.log")

        self.n_steps = self.cfg.n_steps
        self.batch_size = self.cfg.batch_size
        self.target_policy_diff = self.cfg.target_policy_diff
        assert self.batch_size % self.n_steps == 0
        self.buffer_capacity = self.batch_size
        if self.cfg.algo == "sac" or self.cfg.algo == "ddpg":
            self.buffer_capacity = self.batch_size * 8
        self.device = device
        ns_api = NameClient(cfg)

        # net server
        self.zmq_adaptor = ZmqAdaptor(logger=self.logger)
        address, port = ns_api.register(rtype="training_server", extra={"data_type": "train", "zmq_mode": "pull", "team_id": team_id})
        self.ip = "%s_%d" % (address, port)
        self.logger.info(f"data server begin listen {self.ip}")
        self.zmq_adaptor.start({"mode": "pull", "host": "*", "port": port})

        self.data_set = PPODataSet(max_capacity=self.buffer_capacity, device=device)

        # counter
        self.last_put_data_time = time.time()
        self.receive_instance_num = 0

    def sample_data(self, update_times):
        while True:
            socks = dict(self.zmq_adaptor.poller.poll(timeout=1))
            raw_data_list = []
            if self.zmq_adaptor.receiver in socks and socks[self.zmq_adaptor.receiver] == zmq.POLLIN:
                while True:
                    try:
                        data = self.zmq_adaptor.receiver.recv(zmq.NOBLOCK)
                        raw_data_list.append(data)
                    except zmq.ZMQError as e:
                        if type(e) != zmq.error.Again:
                            self.logger.warn("recv zmq {}".format(e))
                        break

                for raw_data in raw_data_list:
                    all_data = pickle.loads(raw_data)
                    data = all_data["instances"]
                    self.receive_instance_num += len(data)
                    for instance in data:
                        self.data_set.append(instance)

                self.data_set.fit_max_size()

            if self.cfg.algo != "sac":
                if len(self.data_set) == self.buffer_capacity:
                    cur_policy_diff = update_times - self.data_set.get_average_update_time()
                    if self.target_policy_diff is not None and cur_policy_diff > self.target_policy_diff:
                        continue
                    # self.logger.info(f"local data server receive data num: {self.receive_instance_num},"
                    #                  f" time: {time.time() - self.last_put_data_time},"
                    #                  f" buffer len :{len(self.data_set)}/{self.buffer_capacity}")
                    st = time.time()

                    batch_data = self.data_set.prepare_for_training()

                    slice_end_time = time.time()
                    batch_data["receive_instance_num"] = self.receive_instance_num
                    self.receive_instance_num = 0
                    # self.logger.info(f"local_data_server_{self.index} prepare batch_data done,"
                    #                  f" slice data time:{slice_end_time - st},"
                    #                  f" policy_diff:{cur_policy_diff}/{self.target_policy_diff}")
                    self.last_put_data_time = time.time()
                    return batch_data
            elif self.cfg.algo == "sac" or self.cfg.algo == "ddpg":
                if len(self.data_set) >= self.buffer_capacity:
                    batch_data = self.data_set.sample(self.batch_size)
                    batch_data["receive_instance_num"] = self.receive_instance_num
                    self.receive_instance_num = 0
                    self.last_put_data_time = time.time()
                    return batch_data

import pickle
import time
import uuid

from drrl.common.utils import setup_logger
from drrl.network.name_client import NameClient
from drrl.network.zmq import ZmqAdaptor
from drrl.network.moni import Moni


class LogClient:

    def __init__(self, cfg, ns_api=None, net=None):
        self.logger = setup_logger("log_server_api.log")
        self.cfg = cfg

        if ns_api is None:
            self.ns_api = NameClient(cfg)
        else:
            self.ns_api = ns_api

        if net is None:
            self._net = ZmqAdaptor(logger=self.logger)
        else:
            self._net = net

        self.uid = str(uuid.uuid1())
        self.moni_data = {}

    def connect(self):
        # net client to log server
        while True:
            _res, log_services = self.ns_api.discovery_log_server(block=True)
            if len(log_services) == 0:
                continue
            log_server = log_services[0]

            self._net.start({"mode": "push", "host": log_server.address, "port": log_server.port, "dest": "logger"})
            break

    def add_moni(self, msg_type, interval_time):
        self.moni_data[msg_type] = Moni(msg_type, interval_time)

    def record(self, msg_type, data):
        self.moni_data[msg_type].record(data)

    def send_moni(self, msg_type):
        assert msg_type in self.moni_data.keys(), f"msg_type: {msg_type} haven't added"

        if time.time() > self.moni_data[msg_type].last_send_time:
            result = self.moni_data[msg_type].result()
            if self.uid is not None:
                result["uid"] = self.uid
            self._net.logger_sender.send(pickle.dumps(result))
            # self.logger.info("send moni to logserver")

import time
import pickle
import random

import zmq

from drrl.common.utils import setup_logger
from drrl.network.zmq import ZmqAdaptor
from drrl.network.name_client import NameClient


class InferenceClient:
    """
    agent.py will use InferenceServerAPI to
        1. connect to inference server.
        2. send instance to inference server and get inference result.
    """

    def __init__(self, cfg, ns_api=None, net=None):
        self.logger = setup_logger("./inference_client.log")
        self.cfg = cfg

        if ns_api is None:
            self.ns_api = NameClient(cfg)
        else:
            self.ns_api = ns_api

        if net is None:
            self._net = ZmqAdaptor(logger=self.logger)
        else:
            self._net = net

    def connect(self):
        # net client to inference server
        while True:
            _, q_services = self.ns_api.discovery_q_server(block=True)
            q_services = list(filter(lambda x: x.extra["data_type"] == "training", q_services))
            if len(q_services) == 0:
                continue

            q = random.choice(q_services)
            self._net.start({"mode": "req", "host": q.address, "port": q.port, "timeout": 2500})
            return q.address, q.port

    def remote_predict(self, instance, retry_times=3):
        """
        instance: see newton.actor.instance.py

        return: (np.array) shape=[batch_size, concatted_result_size]
        policy should concat all result(action, value, neglogp ...)
        """
        req_data = pickle.dumps(instance)
        retry = 0

        while True:
            try:
                self._net.requester.send(req_data)
                result = self._net.requester.recv()
                result = pickle.loads(result)
                if retry > 0:
                    self.logger.warning("reconnect inference server success!")
                return result
            except zmq.Again:
                self.logger.warning("send or recv inference server failed!")
                retry += 1
                self._net.requester.close()
                if retry >= retry_times:
                    return None
                self.connect()
                time.sleep(1)

import uuid

from drrl.common.utils import setup_logger
from drrl.network.utils import host_ip
from drrl.network.evaluation_server import EvaluationServer
from drrl.network.name_client import NameClient
from drrl.network.zmq import ZmqAdaptor


class EvaluationClient:
    """
    runner.py will use EvaluationClient to
        1. query game match task.
        2. report game match result.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.logger = setup_logger("evaluation_client.log")
        self._net = ZmqAdaptor(logger=self.logger)
        self.ns = NameClient(cfg)
        _res, evaluation_servers = self.ns.discovery_evaluation_server(block=True)

        self._net.start({"mode": "req", "host": evaluation_servers[0].address, "port": evaluation_servers[0].port})

        self.uid = str(uuid.uuid1())

    def finish_match(self, first_model_name: str, second_model_name: str, result: float, ep_reward: float, match_type: str) -> str:
        """
        runner will use finish_match send match result back to evaluation server in self-pool mode.
        """
        service_api = EvaluationServer.Req.FINISH_MATCH
        req = {
            "address": host_ip(),
            "uuid": self.uid,
            "first": first_model_name,
            "second": second_model_name,
            "result": result,
            "ep_reward": ep_reward,
            "match_type": match_type
        }

        self._net.send_request_api(service_api.value, req)

        msg = self._net.receive_response_pyobj()
        return msg["res"]

    def query_task(self):
        """
        runner will use query_task to get play_mode to decide which kind of match(vs env, self play or self pool) it should run.

        return: [
            msg["res"]: (str) status result(ok or failed)
            msg["play_mode"]: (str) which kind of match(vs env, self play or self pool) runner should run
            msg["model_name"]: (str) model name if play_mode is self pool else None
            msg["model_config"]: currently None
            msg["model_dict"]: (dict) model parameter if play_mode is self pool else None
        ]
        """
        service_api = EvaluationServer.Req.QUERY_TASK
        req = {"address": host_ip(), "uuid": self.uid}

        self._net.send_request_api(service_api.value, req)

        msg = self._net.receive_response_pyobj()
        # _res, play_mode, model_name, model_config, model_dict
        return msg["res"], msg["play_mode"], msg["model_name"], msg["model_config"], msg["model_dict"]

    def query_eval(self):
        service_api = EvaluationServer.Req.QUERY_EVAL
        req = {"address": host_ip(), "uuid": self.uid}

        self._net.send_request_api(service_api.value, req)

        return self._net.receive_response_pyobj()


    def get_pool_model(self):
        """
        no use
        return: [
            msg["res"]: (str) status result(ok or failed)
            msg["model_name"]: (str) model name store in model pool
            msg["model_dict"]: (dict) model parameter
        ]
        """
        service_api = EvaluationServer.Req.GET_POOL_MODEL
        req = {"address": host_ip(), "uuid": self.uid}

        self._net.send_request_api(service_api.value, req)

        msg = self._net.receive_response_pyobj()
        return msg["res"], msg["model_name"], msg["model_dict"]

    def get_latest_model(self):
        """
        no use
        return: [
            msg["res"]: (str) status result(ok or failed)
            msg["model_dict"]: (dict) latest training model parameter
        ]
        """
        service_api = EvaluationServer.Req.GET_LATEST_MODEL
        req = {"address": host_ip(), "uuid": self.uid}

        self._net.send_request_api(service_api.value, req)

        msg = self._net.receive_response_pyobj()
        return msg["res"], msg["model_dict"]

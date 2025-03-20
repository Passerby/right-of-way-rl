import pickle
import time
from enum import Enum

from drrl.common.utils import setup_logger
from drrl.network.zmq import ZmqAdaptor


class NameServer:
    """ 服务注册与服务发现
    职责: 用于存储服务UUID, IP, 和端口以及服务名称
    第一版本的设计过于依赖 redis 扩展性有待加强, debug 方式也不方便
    """

    class Req(Enum):
        """ 枚举服务请求类型
        """
        REGISTER = "register"
        REGISTER_GPU = "register_gpu"
        DISCOVERY_ALL = "discovery_all"
        DISCOVERY_INFERENCE = "discovery_inference"
        DISCOVERY_TRAIN = "discovery_train"
        DISCOVERY_AGENTS = "discovery_agent"
        DISCOVERY_LOG_SERVER = "discovery_log_server"
        DISCOVERY_PUB_MODEL_SERVER = "discovery_pub_model_server"
        DISCOVERY_EVALUATION_SERVER = "discovery_evaluation_server"
        DISCOVERY_ALL_GPU = "discovery_all_gpu"
        DISCOVERY_AVAILABLE_GPU = "discovery_available_gpu"

        ALL_KEY = "all_keys"
        GET_TASK = "get_task"
        KEEP_ALIVE = "keep_alive"

    class Res(Enum):
        """ 枚举返回相应类型
        """
        OK = "ok"
        INVALID_PARAM = "invalid_param"
        INVALID_API = "invalid_api"
        NOT_FOUND_MODEL = "not_found_model"

    class Service():
        """ 服务
        """

        def __init__(self, address: str, port: int, uuid: str, extra=None):
            self.address = address
            self.port = port
            self.uuid = uuid
            self.extra = extra

            self.ts = time.time()

        def keep_alive(self):
            self.ts = time.time()

        @property
        def data(self):
            return {"address": self.address, "port": self.port, "uuid": self.uuid, "extra": self.extra}

    def __init__(self, cfg) -> None:
        super().__init__()

        self.cfg = cfg

        self._service_table = {}
        self._service_type = {
            "training_server": [],
            "inference_server": [],
            "agent": [],
            "log_server": [],
            "pub_model_server": [],
            "evaluation_server": []
        }

        # 读取 GPU 资源
        self.logger = setup_logger("name_server.log")
        self._gpu_table = {}
        self._uid_gpu_table = {}
        resource_config = cfg.hosts
        self.logger.info("Nameserver loading GPU resources")
        for host in resource_config["gpu_server"]:
            if "gpu_num" in host and "host_ip" in host:
                self._gpu_table[host["host_ip"]] = {}
                for gpu_index in range(host["gpu_num"]):
                    self._gpu_table[host["host_ip"]][gpu_index] = None
                    self.logger.info("Nameserver found GPU {} index: {}".format(host["host_ip"], gpu_index))

        self._net = ZmqAdaptor(logger=self.logger)

        self.clean_up_interval = 180
        self.next_clean_up = time.time() + self.clean_up_interval

        self.qps = {}
        self.qps_check_time = time.time() + 60

    # 启动服务
    def run(self):
        config = self.cfg.name_server
        self._net.start({"mode": "rep", "host": "*", "port": config["port"]})
        self.qps_check_time = time.time() + 60
        self._run_api_server()

    # 关闭服务
    def close(self):
        self._net.rep_receiver.close()

    def _moni_qps(self, api):
        if api not in self.qps:
            self.qps[api] = 0
        self.qps[api] += 1

    def _check_qps(self):
        if time.time() > self.qps_check_time:
            if hasattr(self._net, 'logger_sender'):
                self.qps['msg_type'] = "name_server"
                self._net.send_log(pickle.dumps(self.qps))
            elif len(self._service_type["log_server"]) != 0:
                uid = self._service_type["log_server"][0]
                log_server = self._service_table[uid]
                self._net.start({"mode": "push", "host": log_server.address, "port": log_server.port, "dest": "logger"})

            self.qps = {}
            self.qps_check_time = time.time() + 60

    def _run_api_server(self):
        self.logger.info("start run api server")

        # TODO exit graceful
        while True:
            self._check_qps()
            socks = dict(self._net.poller.poll())
            if self._net.has_rep_data(socks):
                api, msg = self._net.receive_api_request()

                self._moni_qps(api)

                if api == NameServer.Req.REGISTER.value:
                    self._register(msg)
                elif api == NameServer.Req.REGISTER_GPU.value:
                    self._register_gpu(msg)
                elif api == NameServer.Req.KEEP_ALIVE.value:
                    self._keep_alive(msg)
                elif api == NameServer.Req.DISCOVERY_ALL.value:
                    self._discovery_all()
                elif api == NameServer.Req.DISCOVERY_INFERENCE.value:
                    self._discovery_q_server()
                elif api == NameServer.Req.DISCOVERY_TRAIN.value:
                    self._discovery_train_server()
                elif api == NameServer.Req.DISCOVERY_AGENTS.value:
                    self._discovery_agents()
                elif api == NameServer.Req.DISCOVERY_LOG_SERVER.value:
                    self._discovery_log_server()
                elif api == NameServer.Req.DISCOVERY_PUB_MODEL_SERVER.value:
                    self._discovery_pub_model_server()
                elif api == NameServer.Req.DISCOVERY_EVALUATION_SERVER.value:
                    self._discovery_evaluation_server()
                elif api == NameServer.Req.DISCOVERY_ALL_GPU.value:
                    self._discovery_all_gpu(msg)
                elif api == NameServer.Req.DISCOVERY_AVAILABLE_GPU.value:
                    self._discovery_available_gpu(msg)
                else:
                    self._net.send_response_api({"res": NameServer.Res.INVALID_API.value})
            else:
                self._clean_up()

    def _clean_up(self):
        if time.time() > self.next_clean_up:
            removes_uid = []
            for uid, service in self._service_table.items():
                if time.time() > service.ts + self.clean_up_interval:
                    removes_uid.append(uid)

            for uid in removes_uid:
                self._service_table.pop(uid)

                if uid in self._uid_gpu_table:
                    gpu_service = self._uid_gpu_table[uid]
                    address = gpu_service["address"]
                    gpu_index = gpu_service["gpu_index"]

                    # clean up gpu table
                    self._gpu_table[address][gpu_index] = None
                    self._uid_gpu_table.pop(uid)

            self.next_clean_up = time.time() + self.clean_up_interval

    def _register(self, msg: dict):
        if "port" not in msg or "address" not in msg or "uuid" not in msg:
            self._net.send_response_api({"res": NameServer.Res.INVALID_PARAM.value})
        else:
            self._service_table[msg["uuid"]] = NameServer.Service(
                address=msg["address"], port=msg["port"], uuid=msg["uuid"], extra=msg.get("extra"))
            if "rtype" in msg and msg["rtype"] in self._service_type:
                self._service_type[msg["rtype"]].append(msg["uuid"])
            self._net.send_response_api({"res": NameServer.Res.OK.value})

    def _register_gpu(self, msg: dict):
        if "address" not in msg or "uuid" not in msg:
            self._net.send_response_api({"res": NameServer.Res.INVALID_PARAM.value})
        else:
            uid = msg["uuid"]
            address = msg["address"]
            gpu_index = None

            if "gpu_index" in msg:
                gpu_index = msg["gpu_index"]
            else:
                gpu_index = self._available_gpu(ip=address)

            self._gpu_table[address][gpu_index] = uid
            self._uid_gpu_table[uid] = {"address": address, "gpu_index": gpu_index}

            self._net.send_response_api({"res": NameServer.Res.OK.value, "data": gpu_index})

    def _keep_alive(self, msg: dict):
        if "uuid" not in msg:
            self._net.send_response_api({"res": NameServer.Res.INVALID_PARAM.value})
        elif msg["uuid"] not in self._service_table:
            self._register(msg)
        else:
            self._service_table[msg["uuid"]].keep_alive()
            self._net.send_response_api({"res": NameServer.Res.OK.value})

    def _discovery_all(self):
        data = []
        for _, value in self._service_table.items():
            data.append(value)

        self._net.send_response_api({"res": NameServer.Res.OK.value, "data": data})

    def _discovery_q_server(self):
        self._discovery_rtype_server(rtype="inference_server")

    def _discovery_train_server(self):
        self._discovery_rtype_server(rtype="training_server")

    def _discovery_agents(self):
        self._discovery_rtype_server(rtype="agent")

    def _discovery_log_server(self):
        self._discovery_rtype_server(rtype="log_server")

    def _discovery_pub_model_server(self):
        self._discovery_rtype_server(rtype="pub_model_server")

    def _discovery_evaluation_server(self):
        self._discovery_rtype_server(rtype="evaluation_server")

    def _discovery_rtype_server(self, rtype):
        data = []
        for uid in self._service_type[rtype]:
            if uid not in self._service_table:
                continue
            data.append(self._service_table[uid])

        self._net.send_response_api({"res": NameServer.Res.OK.value, "data": data})

    def _discovery_all_gpu(self, msg):
        if "address" not in msg:
            self._net.send_response_api({"res": NameServer.Res.INVALID_PARAM.value})
        else:
            all_gpu = self._all_gpu(ip=msg["address"])
            self._net.send_response_api({"res": NameServer.Res.OK.value, "data": all_gpu})

    def _discovery_available_gpu(self, msg):
        if "address" not in msg:
            self._net.send_response_api({"res": NameServer.Res.INVALID_PARAM.value})
        else:
            gpu_index = self._available_gpu(ip=msg["address"])
            self._net.send_response_api({"res": NameServer.Res.OK.value, "data": gpu_index})

    def _all_gpu(self, ip):
        return self._gpu_table.get(ip)

    def _available_gpu(self, ip) -> int:
        gpu = self._all_gpu(ip)
        for gpu_index, value in gpu.items():
            if value is None:
                return gpu_index

import zmq

from drrl.common.utils import setup_logger


class ZmqAdaptor:

    def __init__(self, logger=setup_logger("zmq.log")):
        # 因为共用 poller，里面实现各种通信方式,不同方式也不保证唯一
        # 比如既向 gpu server push数据，也向 logger server push 数据
        self.logger = logger
        self.context = zmq.Context()

        self.context.setsockopt(zmq.TCP_KEEPALIVE, 1)
        self.context.setsockopt(zmq.TCP_KEEPALIVE_CNT, 60)
        self.context.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 60)
        self.context.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 60)
        self.poller = zmq.Poller()
        self.modes = []

    def start(self, config):
        self.modes.append(config["mode"])
        if config["mode"] == "pull":
            self.receiver = self.context.socket(zmq.PULL)
            self.receiver.bind("tcp://%s:%d" % (config["host"], config["port"]))
            self.poller.register(self.receiver, zmq.POLLIN)
        elif config["mode"] == "push":
            if "dest" in config and config["dest"] == "logger":
                self.logger_sender = self.context.socket(zmq.PUSH)
                self.logger_sender.connect("tcp://%s:%d" % (config["host"], config["port"]))
            else:
                self.sender = self.context.socket(zmq.PUSH)
                self.sender.connect("tcp://%s:%d" % (config["host"], config["port"]))
        elif config["mode"] == "req":
            self.requester = self.context.socket(zmq.REQ)
            if config.get("timeout") is not None:
                self.requester.setsockopt(zmq.SNDTIMEO, config.get("timeout"))
                self.requester.setsockopt(zmq.RCVTIMEO, config.get("timeout"))
            self.requester.connect("tcp://%s:%d" % (config["host"], config["port"]))
        elif config["mode"] == "router":
            self.router_receiver = self.context.socket(zmq.ROUTER)
            self.router_receiver.bind("tcp://%s:%d" % (config["host"], config["port"]))
            self.poller.register(self.router_receiver, zmq.POLLIN)
        elif config["mode"] == "sub":
            self.subscriber = self.context.socket(zmq.SUB)
            self.subscriber.connect("tcp://%s:%d" % (config["host"], config["port"]))
            self.subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
            self.poller.register(self.subscriber, zmq.POLLIN)
        elif config["mode"] == "pub":
            self.publisher = self.context.socket(zmq.PUB)
            self.publisher.bind("tcp://%s:%d" % (config["host"], config["port"]))
        elif config["mode"] == "rep":
            self.rep_receiver = self.context.socket(zmq.REP)
            self.rep_receiver.bind("tcp://%s:%d" % (config["host"], config["port"]))
            self.poller.register(self.rep_receiver, zmq.POLLIN)
        else:
            self.logger.warn("wrong mode {0}".format(config["mode"]))
        self.logger.debug("mode {0}, host {1} port {2}".format(config["mode"], config["host"], config["port"]))

    def send_log(self, data):
        self.logger_sender.send(data)

    def push(self, data: bytes):
        self.sender.send(data)

    def receive(self) -> bytes:
        return self.receiver.recv(zmq.NOBLOCK)

    def push_pyobj(self, data):
        self.sender.send_pyobj(data)

    def receive_pyobj(self):
        return self.receiver.recv_pyobj(zmq.NOBLOCK)

    def poll(self, timeout=100) -> dict:
        return dict(self.poller.poll(timeout=timeout))

    def receive_request(self):
        return self.router_receiver.recv_multipart(zmq.NOBLOCK, copy=False)

    def receive_response(self):
        return self.requester.recv()

    def receive_response_pyobj(self):
        return self.requester.recv_pyobj()

    def send_request(self, data):
        self.requester.send(data)

    def send_request_pyobj(self, data):
        self.requester.send_pyobj(data)

    def send_response(self, data):
        self.router_receiver.send_multipart(data)

    def send_request_api(self, api, data):
        self.requester.send_string(api, zmq.SNDMORE)
        self.requester.send_pyobj(data)

    def receive_api_request(self):
        api = self.rep_receiver.recv_string()
        msg = self.rep_receiver.recv_pyobj()
        return api, msg

    def send_response_api(self, data):
        self.rep_receiver.send_pyobj(data)

    def has_rep_data(self, socks) -> bool:
        return self.rep_receiver in socks and socks[self.rep_receiver] == zmq.POLLIN

    def has_data(self, socks) -> bool:
        return self.receiver in socks and socks[self.receiver] == zmq.POLLIN

    def has_request(self, socks) -> bool:
        return self.router_receiver in socks and socks[self.router_receiver] == zmq.POLLIN

    def pub(self, data):
        self.publisher.send(data)

    def has_publish_data(self, socks) -> bool:
        return self.subscriber in socks and socks[self.subscriber] == zmq.POLLIN

    def sub(self):
        return self.subscriber.recv()

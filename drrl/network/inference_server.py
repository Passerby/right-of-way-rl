import pickle
import time

import zmq
import torch

from drrl.common.utils import setup_logger, is_development
from drrl.network.zmq import ZmqAdaptor
from drrl.network.name_client import NameClient
from drrl.network.log_client import LogClient
# from drrl.models.policies.multi_team_policy import MultiTeamPolicy
from drrl.models.policies.ctde_team_policy import CTDETeamPolicy
from drrl.models.policies.ctce_team_policy import CTCETeamPolicy
from drrl.models.policies.sac_policy import SACPolicy
from drrl.models.policies.ddpg_policy import DDPGPolicy
from drrl.data.meta_drive_ppo_dataset import MetaDrivePPODataSet as PPODataSet


class InferenceServer:

    def __init__(self, cfg, device="auto", inference_type="training", sep_policy=False):
        self.test = is_development()
        self.device = device
        self.team_id = None
        if not torch.cuda.is_available() and self.device == "gpu":
            self.device = "cpu"

        if sep_policy:
            if device == "cuda:2":
                self.team_id = "first_0"
            elif device == "cuda:3":
                self.team_id = "second_0"

        self.inference_type = inference_type # training or fixed

        self.player_num = cfg['policy']['placeholder']['max_agents']

        self.logger = setup_logger("./inference_server_log_{0}.log".format(self.inference_type))

        # net server
        self.zmq_adaptor = ZmqAdaptor(logger=self.logger)
        ns_api = NameClient(cfg)
        self.logger.info("begin register inference server to name server")
        address, port = ns_api.register(
            rtype="inference_server", extra={
                "data_type": self.inference_type,
                "zmq_mode": "router",
                "team_id": self.team_id
            })
        self.zmq_adaptor.start({"mode": "router", "host": "*", "port": port})
        self.ip = "%s_%d" % (address, port)
        self.logger.info(f"begin listen {self.ip}")

        # net clinet to publish model server
        if self.inference_type == "training":
            self.logger.info("need found pub model server")
            while True:
                pub_services = None
                _res, pub_services = ns_api.discovery_pub_model_server(block=True)
                # fileter pub model server team_id
                pub_services = list(filter(lambda x: x.extra['team_id'] == self.team_id, pub_services))
                if len(pub_services) > 0:
                    self.zmq_adaptor.start({"mode": "sub", "host": pub_services[0].address, "port": pub_services[0].port})
                    break
                else:
                    self.logger.info("not found pub model server, try again.")
                time.sleep(1)

        # net client to log server
        self.ls_api = LogClient(cfg, ns_api=ns_api, net=self.zmq_adaptor)
        self.ls_api.connect()
        self.ls_api.add_moni(msg_type="inference_server", interval_time=5)

        # self.gpu_num = 0
        # if self.test:
        #     self.logger.info('debug model')
        if self.device == "cpu":
            torch.set_num_threads(8)
            self.logger.info('device cpu model')
        else:
            self.logger.info('device gpu model, gpu num: %s' % (self.device))

        if cfg.algo == 'ppo':
            self.policy = CTDETeamPolicy(cfg).to(self.device)
            # self.policy = CTCETeamPolicy(cfg).to(self.device)
        elif cfg.algo == 'sac':
            self.policy = SACPolicy(cfg).to(self.device)
        elif cfg.algo == 'ddpg':
            self.policy = DDPGPolicy(cfg).to(self.device)
        self.data_set = PPODataSet(max_capacity=10000, data_type="inference_type", device=self.device)

        self.max_waiting_time = 0.010
        self.next_check_time = time.time()
        self.raw_data_list = []

    def receive_model(self, socks):
        model_parameter = None
        if self.inference_type == "training":
            # receive model parameter
            if self.zmq_adaptor.subscriber in socks and socks[self.zmq_adaptor.subscriber] == zmq.POLLIN:
                model_parameter = self.zmq_adaptor.subscriber.recv()

            # deserialize model
            if model_parameter is not None:
                bt = time.time()
                model_dict = pickle.loads(model_parameter)
                new_model_dict = {}
                for k, v in model_dict.items():
                    if k.startswith("module."):
                        new_model_dict[k[7:]] = v.to(self.device)
                    else:
                        new_model_dict[k] = v.to(self.device)

                del model_dict

                self.policy.load_state_dict(new_model_dict)
                self.policy.to(self.device)

                self.model_time = self.policy.update_time

                model_delay_t = (time.time() - self.model_time) * 1000
                load_model_dt = (time.time() - bt) * 1000
                # self.logger.info("load model {0}, use time {1} ms, model delay {2} ms".format(self.policy.update_time, load_model_dt,
                #                                                                         model_delay_t))
                moni_dict = {"model_delay_time": model_delay_t, "inf_server_load_model_time": load_model_dt}
                self.ls_api.record(msg_type="inference_server", data=moni_dict)
        elif self.inference_type == "fixed":
            pass

    def receive_data(self, socks):
        if self.zmq_adaptor.router_receiver in socks and socks[self.zmq_adaptor.router_receiver] == zmq.POLLIN:
            receive_data_count = 0
            tmp_list = []
            while True:
                try:
                    data = self.zmq_adaptor.router_receiver.recv_multipart(zmq.NOBLOCK)
                    tmp_list.append(data)
                except zmq.ZMQError as _:
                    break
            # if len(tmp_list) > 0:
            #     self.logger.info('recieve data num {0}'.format(len(tmp_list)))
            for raw_data in tmp_list:
                self.raw_data_list.append(raw_data)
                instance = pickle.loads(raw_data[-1])
                self.data_set.append(instance)
                receive_data_count += 1
            self.ls_api.record(msg_type="inference_server", data={"inf_server_receive_instance": receive_data_count})

    def run(self):
        socks = dict(self.zmq_adaptor.poller.poll())
        self.receive_model(socks)
        self.logger.info("inference_server latest model ready.")
        while True:
            socks = dict(self.zmq_adaptor.poller.poll(timeout=0.025))
            self.receive_model(socks)
            self.receive_data(socks)
            cur_size = len(self.data_set)
            if cur_size > 0 and time.time() > self.next_check_time:
                self.next_check_time = time.time() + self.max_waiting_time
                s_time = time.time()
                # self.logger.info("start convert_to_np, count {0}, gpu {1}".format(cur_size, self.device))
                data = self.data_set.prepare_for_inference()
                # self.logger.info("start get_predict_action_batch, cost {0}".format(time.time() - s_time))
                # results = action, value, state_out, neglogpac
                raw_result = self.policy.batch_step(data)

                result_list = raw_result.reshape((-1, self.player_num, raw_result.shape[-1]))
                result_list = result_list.detach().cpu().numpy()
                for index, m in enumerate(self.raw_data_list):
                    m[-1] = pickle.dumps(result_list[index])
                    self.zmq_adaptor.router_receiver.send_multipart(m)
                self.raw_data_list = []
                self.data_set.clear()
                # self.logger.info("send back finish, cost: {0}".format(time.time() - s_time))
            self.ls_api.send_moni(msg_type="inference_server")

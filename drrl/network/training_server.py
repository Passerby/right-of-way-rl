import sys
import time
import pickle

import torch
import numpy as np

from drrl.common.utils import setup_logger, is_development, create_path, size_format
from drrl.models.algorithms.ppo import PPO
from drrl.models.algorithms.sac import SAC
from drrl.models.algorithms.ddpg import DDPG
from drrl.models.utils import save_ckp_model, save_optimizer
from drrl.network.zmq import ZmqAdaptor
from drrl.network.name_client import NameClient
from drrl.network.data_client import DataClient
from drrl.network.log_client import LogClient
# from drrl.models.policies.multi_team_policy import MultiTeamPolicy
from drrl.models.policies.ctde_team_policy import CTDETeamPolicy
from drrl.models.policies.ctce_team_policy import CTCETeamPolicy
from drrl.models.policies.sac_policy import SACPolicy
from drrl.models.policies.ddpg_policy import DDPGPolicy

class TrainingServer:

    def __init__(self, cfg, rank=0, use_distribution=False, sep_policy=False):
        self.cfg = cfg
        self.test = is_development()
        self.rank = rank
        self.mpi_rank = rank
        self.device = "cuda:%d" % rank
        self.team_id = None
        self.sep_policy = sep_policy

        self.batch_size = cfg.batch_size
        self.model_save_freq = cfg.model_save_freq
        self.exp_name = cfg.log_server["exp_name"]

        if sep_policy:
            if rank == 0:
                self.team_id = "first_0"
            elif rank == 1:
                self.team_id = "second_0"

        pull_ns_api = NameClient(cfg)


        # with gpu num 0
        if not torch.cuda.is_available():
            self.device = "cpu"

        self.checkpoint_dir = "./checkpoints" if self.rank == 0 else None

        need_dirs = [
            "./log/saved_model_%s_%d/" % (self.exp_name, self.mpi_rank),
            "./log/gpu_server_log_%s_%d/" % (self.exp_name, self.mpi_rank)
        ]
        for d in need_dirs:
            create_path(d)
        self.logger = setup_logger("gpu_server_log_%s_%d.log" % (self.exp_name, self.mpi_rank))

        # net server
        self.zmq_adaptor = ZmqAdaptor(logger=self.logger)

        # publish model
        if self.rank == 0 or self.sep_policy:
        # log self.team_id
            self.logger.info("begin register pub model server to name server {}".format(self.team_id))
            ns_api = NameClient(cfg)
            _, port = ns_api.register(rtype="pub_model_server", extra={"data_type": "train", "zmq_mode": "pub", "team_id": self.team_id})
            self.zmq_adaptor.start({"mode": "pub", "host": "*", "port": port})

        # net client to log server
        self.ls_api = LogClient(cfg, ns_api=pull_ns_api, net=self.zmq_adaptor)
        self.ls_api.connect()

        self.ds_api = DataClient(cfg)
        self.ds_api.connect(self.mpi_rank, self.device, self.team_id)

        if cfg.algo == 'ppo':
            # self.policy = CTCETeamPolicy(cfg).to(self.device)
            self.policy = CTDETeamPolicy(cfg).to(self.device)
        elif cfg.algo == 'sac':
            self.policy = SACPolicy(cfg).to(self.device)
        elif cfg.algo == 'ddpg':
            self.policy = DDPGPolicy(cfg).to(self.device)

        if use_distribution:
            self.policy = torch.nn.parallel.DistributedDataParallel(self.policy, device_ids=[self.device])

        if cfg.algo == 'ppo':
            self.model = PPO(cfg, self.policy)
        elif cfg.algo == 'sac':
            self.model = SAC(cfg, self.policy)
        elif cfg.algo == 'ddpg':
            self.model = DDPG(cfg, self.policy)

        self.sess = None # tf.Session()
        self.last_sgd_time = time.time()
        self.start_training = False

        #  Moni
        self.next_save_model_time = time.time() + 60
        self.model_time = 0 # 保存模型的时间戳
        self.next_check_stat_time = time.time() + 60
        self.receive_instance_num = 0
        self.ls_api.add_moni(msg_type="training_server", interval_time=10)
        self.ls_api.add_moni(msg_type="model", interval_time=0)

    def run(self):

        if self.rank == 0 or self.sep_policy:
            self.pub_model()

        training_end_time = time.time()
        while True:
            if time.time() > self.next_save_model_time: # and self.rank == 0:
                self.save_model()
                if self.test:
                    self.next_save_model_time += 60 * 2
                else:
                    self.next_save_model_time += 60 * self.model_save_freq
            batch_data, wait_data_server_count = self.ds_api.sample_data(
                update_times=self.model.update_count, device=self.device)
            # self.logger.info(f"training server got batch data and delete object index {self.mpi_rank}")
            self.ls_api.record(msg_type="training_server", data={"receive_data_time": time.time() - training_end_time})
            self.learn(batch_data)
            self.ls_api.record(
                msg_type="training_server",
                data={
                    "total_training_time": time.time() - training_end_time,
                    "wait_data_server_count": wait_data_server_count
                },
            )
            training_end_time = time.time()
            self.ls_api.send_moni(msg_type="training_server")
            self.ls_api.send_moni(msg_type="model")

    def learn(self, batch_data):
        receive_instance_num = batch_data.pop("receive_instance_num")
        if type(batch_data["model_update_times"]) == torch.Tensor:
            policy_different = self.model.update_count - torch.mean(batch_data["model_update_times"]).item()
        else:
            policy_different = self.model.update_count - np.mean(batch_data["model_update_times"])
        start_time = time.time()
        model_log_dict, train_log_dict = self.model.learn(batch_data)
        end_time = time.time()
        # self.logger.info("total sgd time: {0}".format(end_time - start_time))

        self.model.increment_update()

        if self.rank == 0 or self.sep_policy:
            self.pub_model()

        if self.cfg.algo == 'ppo':
            model_log_dict.update({
                "Q_value": torch.mean(batch_data["returns"]).item(),
                "advantage_value": torch.mean(batch_data["adv"]).item(),
            })

        self.ls_api.record(msg_type="model", data=model_log_dict)

        train_log_dict.update({
            "training_steps_total": self.model.update_count,
            "policy_different": policy_different,
            # "action_entropy_parameter": self.model.ent_coef,
            "learning_rate": self.model.learning_rate,
            "receive_instance_total": receive_instance_num,
            f"receive_instance_rank_{self.rank}": receive_instance_num,
            "data_efficiency": receive_instance_num / self.batch_size,
            "total_sgd_time": end_time - start_time,
        })

        self.ls_api.record(msg_type="training_server", data=train_log_dict)

    def pub_model(self):
        state_dict_cpu = self.policy.state_dict()
        model_bytes = pickle.dumps(state_dict_cpu)

        # self.logger.debug("root gpu server pub model, model size {0},"
        #                   " mode {1}".format(size_format(model_bytes), self.exp_name))

        # 只通过 zmq 传给 worker gpu
        self.zmq_adaptor.publisher.send(model_bytes)

    def save_model(self):
        bt = time.time()

        save_ckp_model(self.policy, checkpoint_path="./log/saved_model_{0}_{1}/".format(self.exp_name, self.mpi_rank))
        if self.cfg.algo == 'ppo':
            save_optimizer(self.model.optimizer, checkpoint_path="./log/saved_model_{0}_{1}/".format(self.exp_name, self.mpi_rank))

        self.logger.info("save model, use {0}".format(time.time() - bt))

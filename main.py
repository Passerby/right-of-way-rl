from multiprocessing import Process
import os
import time

import torch
import hydra

from drrl.actors.meta_drive_runner import MetaDriveRunner
from drrl.network.name_server import NameServer
from drrl.network.training_server import TrainingServer
from drrl.network.log_server import LogServer
from drrl.network.inference_server import InferenceServer
from drrl.network.evaluation_server import EvaluationServer


def runner(cfg):
    runner = MetaDriveRunner(cfg)
    runner.run()

def run_name_server(cfg):
    name_server = NameServer(cfg)
    name_server.run()

def run_log_server(cfg):
    log_server = LogServer(cfg)
    log_server.run()


def run_training(cfg):
    training_server = TrainingServer(cfg)
    training_server.run()

def run_evaluation(cfg):
    evaluation_server = EvaluationServer(cfg)
    evaluation_server.run()

def run_inference(cfg, rank=3):
    # inference_server = InferenceServer(cfg, device="cuda:{}".format(rank))
    inference_server = InferenceServer(cfg, device="cpu")
    inference_server.run()

@hydra.main(config_path="./drrl/configs", config_name="ppo", version_base="1.3.0")
def main(cfg):
    torch.multiprocessing.set_start_method('spawn')
    # setup config placeholder

    # new process for name client
    p = Process(target=run_name_server, args=(cfg,))
    p.start()

    p2 = Process(target=run_log_server, args=(cfg,))
    p2.start()

    p3 = Process(target=run_training, args=(cfg,))
    p3.start()

    for i in range(8):
        p3 = Process(target=run_inference, args=(cfg, 8 - i))
        p3.start()

    time.sleep(3)

    p4 = []
    for _ in range(128):
        sp = Process(target=runner, args=(cfg,))
        p4.append(sp)
        sp.start()

    runner(cfg)

    p.kill()
    p2.kill()
    p3.kill()
    for sp in p4:
        sp.kill()


if __name__ == "__main__":
    print('start running ...')
    main()

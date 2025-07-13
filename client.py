from multiprocessing import Process
import time
import hydra

from drrl.actors.meta_drive_runner import MetaDriveRunner

def runner(cfg):
    runner = MetaDriveRunner(cfg)
    runner.run()


@hydra.main(config_path="./drrl/configs", config_name="ppo", version_base="1.3.0")
def main(cfg):
    processes = []
    for _ in range(128):
        sp = Process(target=runner, args=(cfg,))
        processes.append(sp)
        sp.start()

    for sp in processes:
        sp.kill()


if __name__ == "__main__":
    print('start server running ...')
    main()

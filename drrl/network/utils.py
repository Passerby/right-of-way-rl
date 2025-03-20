import random
import socket
from sys import platform

from drrl.common.utils import is_development


def host_ip() -> str:
    if is_development():
        if platform == "win32":
            # windows 开发环境
            return "127.0.0.1"
        else:
            # linux or docker 开发环境
            return "0.0.0.0"
    else:
        return socket.gethostbyname(socket.getfqdn(socket.gethostname()))
        # some bugs in dockers
        # return '0.0.0.0'


# 随机选取本地可用端口
def random_port() -> int:
    while True:
        port = random.randint(10000, 14000)
        if is_local_open(port):
            return port


# 测试本定端口是否被占用
def is_local_open(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host_ip(), port))
        return True
    except Exception:
        return False
    finally:
        sock.close()

def cal_elo(left_elo: float, right_elo: float, result: float, elo_k: float):
    e_left = 1 / (1 + 10**((right_elo - left_elo) / 400))
    e_right = 1 - e_left

    k = elo_k

    # 高端局递减
    if left_elo > 2000 or right_elo > 2000:
        k /= 2

    left_elo += k * (result - e_left)
    right_elo += k * ((1 - result) - e_right)

    return left_elo, right_elo

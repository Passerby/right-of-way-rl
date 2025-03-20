import os
import sys
import importlib
import logging
import shutil


def setup_logger(logger_name: str = "drrl.log"):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    # stream_handler = logging.StreamHandler()
    # stream_handler.setLevel(logging.DEBUG)
    # stream_handler.setFormatter(formatter)

    # # 文件日志处理器
    file_handler = logging.FileHandler(logger_name)  # 指定日志文件的路径
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def compute_gae_reward(instances, discount_rate, gae_lam, next_value=None, next_done=None):
    last_gae = 0
    for index in reversed(range(len(instances))):
        if index == len(instances) - 1:
            next_state_value = next_value
        else:
            next_state_value = instances[index + 1].value
            next_done = instances[index + 1].done
        item = instances[index]
        nextnonterminal = 1.0 - next_done
        delta = item.reward + discount_rate * next_state_value * nextnonterminal - item.value
        last_gae = delta + gae_lam * discount_rate * nextnonterminal * last_gae
        instances[index].adv = last_gae
        instances[index].returns = last_gae + item.value
    return instances


def create_path(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def size_format(o):
    """
    Converts a size in bytes to a more human-readable format (KB, MB, GB, etc.)

    :param size_in_bytes: Size in bytes
    :return: Size in human-readable format
    """
    size_in_bytes = sys.getsizeof(o)
    if size_in_bytes < 1024:
        return f"{size_in_bytes} bytes"
    elif size_in_bytes < 1024**2:
        return f"{size_in_bytes / 1024:.2f} KB"
    elif size_in_bytes < 1024**3:
        return f"{size_in_bytes / (1024**2):.2f} MB"
    else:
        return f"{size_in_bytes / (1024**3):.2f} GB"


# 示例使用


def environment():
    return "production" if is_production() else "development"


# 判断是否是生产环境
def is_production():
    return not is_development()


# 判断是否是开发环境
def is_development():
    return "LOCALTEST" in os.environ.keys()


def import_module_or_data(import_path):
    try:
        maybe_module, maybe_data_name = import_path.rsplit(".", 1)
        return getattr(importlib.import_module(maybe_module), maybe_data_name)
    except Exception as _:
        try:
            return importlib.import_module(import_path)
        except Exception as e:
            print('Cannot import module, error {}'.format(str(e)))

    raise ImportError('Cannot import module or data using {}'.format(import_path))

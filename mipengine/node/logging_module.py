import logging
import sys
import time

from functools import wraps

from mipengine.node import config as node_config


def getLogger(name):
    logger = logging.getLogger(name)
    formatter = logging.Formatter(
        "%(asctime)s -"
        "%(levelname)s - "
        "NODE - "
        f"{node_config.role} - "
        f"{node_config.identifier} - "
        "%(name)s - "
        "%(funcName)s(%(lineno)d) - "
        "%(message)s"
    )

    # adding formatting handler to the file handler NOT the logger
    # do it in stdout

    handler = logging.StreamHandler(sys.stdout)
    logger.setLevel(node_config.log_level)
    # adding formatting handler to the file handler NOT the logger
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def log_function_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = getLogger(func.__name__)
        starting_timestamp =time.time()
        logger.info(f"*********** {func.__name__} task started ***********")
        output = func(*args, **kwargs)
        finish_timestamp = time.time()
        tsm_diff = finish_timestamp - starting_timestamp
        logger.info(f"*********** {func.__name__} task succeeded in {tsm_diff} ***********")
        return output

    return wrapper

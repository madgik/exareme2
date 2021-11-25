import logging
import sys
import time
from functools import wraps
import quart.logging as ctrl_logging
from mipengine.controller import config as ctrl_config


def getLogger(name):
    logger = ctrl_logging.getLogger(name)
    formatter = ctrl_logging.Formatter(
        "%(asctime)s - "
        "%(levelname)s - "
        "CONTROLLER - "
        "%(name)s - "
        "%(funcName)s(%(lineno)d) - "
        "%(message)s"
    )

    handler = ctrl_logging.StreamHandler(sys.stdout)
    logger.setLevel(ctrl_config.log_level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def log_method_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = getLogger(func.__name__)
        starting_timestamp = time.time()
        logger.info(f"*********** {func.__name__} method started ***********")
        output = func(*args, **kwargs)
        finish_timestamp = time.time()
        tsm_diff = finish_timestamp - starting_timestamp
        logger.info(
            f"*********** {func.__name__} method succeeded in {tsm_diff} ***********"
        )
        return output

    return wrapper

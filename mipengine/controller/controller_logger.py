import logging
import time
from functools import wraps


def getLogger():
    return logging.getLogger("quart.serving")


def log_method_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = getLogger()
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

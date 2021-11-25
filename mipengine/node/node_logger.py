import time
from functools import wraps

from celery.utils.log import get_task_logger


def getLogger(name):
    return get_task_logger(name)


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

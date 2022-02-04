import inspect
import logging
import time

from functools import wraps
from mipengine.node import config as node_config
from mipengine.node_exceptions import ContextIDNotFound


def init_logger(request_id):
    logger = logging.getLogger("node")

    formatter = logging.Formatter(
        f"%(asctime)s - %(levelname)s - NODE - {node_config.role} - {node_config.identifier} - %(module)s - %(funcName)s(%(lineno)d) - {request_id} - %(message)s"
    )

    # StreamHandler
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    for hdlr in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(hdlr)
    logger.addHandler(sh)
    logger.setLevel(node_config.log_level)
    logger.propagate = False

    return logger


def get_logger():
    return logging.getLogger("node")


def initialise_logger(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        arglist = inspect.getfullargspec(func)
        if kwargs.get("request_id"):
            request_id = kwargs.get("request_id")
        elif "request_id" in arglist.args:
            # finds the index of request_id arg in list of args from inspect
            # and finds values in args list
            request_id_index = arglist.args.index("request_id")
            request_id = args[request_id_index]
        else:
            raise ContextIDNotFound()

        init_logger(request_id)
        return func(*args, **kwargs)

    return wrapper


def log_method_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger()
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

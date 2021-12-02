import inspect
import logging
import time

from functools import wraps
from mipengine.node import config as node_config


def init_logger(context_id=None):
    logger = logging.getLogger("node")
    if context_id is None:
        formatter = logging.Formatter(
            f"%(asctime)s - %(levelname)s - NODE - {node_config.role} - {node_config.identifier} - %(module)s - %(funcName)s(%(lineno)d) - %(message)s"
        )
    else:
        formatter = logging.Formatter(
            f"%(asctime)s - %(levelname)s - NODE - {node_config.role} - {node_config.identifier} - %(module)s - %(funcName)s(%(lineno)d) - {context_id} - %(message)s"
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


# TODO: All shared tasks should pass context_id. Relevant ticket: https://team-1617704806227.atlassian.net/browse/MIP-477
def initialise_logger(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        arglist = inspect.getfullargspec(func)
        if kwargs.get("context_id"):
            context_id = kwargs.get("context_id")
        elif "context_id" in arglist.args:
            # finds the index of context_id arg in list of args from inspect
            # and finds values in args list
            context_id_index = arglist.args.index("context_id")
            print("context_id_index")
            context_id = args[context_id_index]
        else:
            context_id = None

        init_logger(context_id)
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

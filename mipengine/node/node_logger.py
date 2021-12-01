import inspect
import logging
import time
from functools import wraps

from mipengine.node import config as node_config


def init_logger(ctx_id=None):
    logger = logging.getLogger("node")
    if ctx_id == None:
        formatter = logging.Formatter(f"%(message)s")
    else:
        formatter = logging.Formatter(
            f"%(asctime)s - %(levelname)s - NODE - ctx_id: {ctx_id} - {node_config.role} - {node_config.identifier} - %(name)s - %(message)s"
        )

    # StreamHandler
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    for hdlr in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(hdlr)
    logger.addHandler(sh)
    logger.setLevel(node_config.log_level)

    return logger


def get_logger():
    return logging.getLogger("node")


def log_add_ctx_id(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        arglist = inspect.getfullargspec(func)
        if kwargs.get("context_id"):
            ctx_id = kwargs.get("context_id")
        elif "context_id" in arglist.args:
            # finds the index of context_id arg in list of args from inspect
            # and finds values in args list
            ctx_idIndex = arglist.args.index("context_id")
            ctx_id = args[ctx_idIndex]
        else:
            ctx_id = None

        init_logger(ctx_id)
        func(*args, **kwargs)

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

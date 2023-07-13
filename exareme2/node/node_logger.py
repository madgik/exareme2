import inspect
import logging
import time
from functools import wraps

from celery import current_task

from exareme2.exceptions import RequestIDNotFound
from exareme2.node import config as node_config

LOGGING_ID_TASK_PARAM = "request_id"

task_loggers = {}


def init_logger(request_id):
    logger = logging.getLogger(request_id)

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
    return task_loggers[current_task.request.id]


def initialise_logger(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        arglist = inspect.getfullargspec(func)
        if kwargs.get(LOGGING_ID_TASK_PARAM):
            request_id = kwargs.get(LOGGING_ID_TASK_PARAM)
        elif LOGGING_ID_TASK_PARAM in arglist.args:
            # finds the index of request_id arg in list of args from inspect
            # and finds values in args list
            request_id_index = arglist.args.index(LOGGING_ID_TASK_PARAM)
            request_id = args[request_id_index]
        else:
            raise RequestIDNotFound()

        task_loggers[current_task.request.id] = init_logger(request_id)
        function = func(*args, **kwargs)
        del task_loggers[current_task.request.id]
        return function

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

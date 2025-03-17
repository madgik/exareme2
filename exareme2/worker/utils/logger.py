import inspect
import logging
import time
from functools import wraps

from celery import current_task
from pythonjsonlogger import jsonlogger

from exareme2.worker import config as worker_config
from exareme2.worker_communication import RequestIDNotFound

LOGGING_ID_TASK_PARAM = "request_id"
task_loggers = {}


# Custom JSON Formatter for task logs
class TaskJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        log_record["federation"] = worker_config.federation
        log_record["worker_role"] = f"exareme2-{worker_config.role.lower()}"
        log_record["worker_identifier"] = worker_config.identifier
        # We set the logger's name to the request_id
        log_record["request_id"] = record.name


def init_logger(request_id):
    logger = logging.getLogger(request_id)
    formatter = TaskJsonFormatter(
        "%(asctime)s %(levelname)s %(module)s %(funcName)s %(lineno)d %(message)s"
    )
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    # Remove existing handlers if any
    for hdlr in logger.handlers[:]:
        logger.removeHandler(hdlr)
    logger.addHandler(sh)
    logger.setLevel(worker_config.log_level)
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
            request_id_index = arglist.args.index(LOGGING_ID_TASK_PARAM)
            request_id = args[request_id_index]
        else:
            raise RequestIDNotFound()
        task_loggers[current_task.request.id] = init_logger(request_id)
        result = func(*args, **kwargs)
        del task_loggers[current_task.request.id]
        return result

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

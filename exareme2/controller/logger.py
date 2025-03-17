import logging

from pythonjsonlogger import jsonlogger

from exareme2.controller import BACKGROUND_LOGGER_NAME
from exareme2.controller import config as ctrl_config


def get_request_logger(request_id):
    """
    Used for logging information produced after an endpoint request.
    """
    logger = (
        init_logger(request_id)
        if not logging.getLogger(request_id).hasHandlers()
        else logging.getLogger(request_id)
    )
    return logger


def init_logger(request_id, log_level=None):
    logger = logging.getLogger(request_id)
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(module)s %(funcName)s %(lineno)d "
        "federation=%(federation)s node_identifier=%(node_identifier)s request_id=%(request_id)s "
        "message=%(message)s"
    )

    # StreamHandler
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    if log_level:
        logger.setLevel(log_level)
    else:
        logger.setLevel(ctrl_config.log_level)

    # Extra attributes for JSON logging
    extra = {
        "federation": ctrl_config.federation,
        "node_identifier": ctrl_config.node_identifier,
        "request_id": request_id,
    }
    logger = logging.LoggerAdapter(logger, extra)

    return logger


def get_background_service_logger() -> logging.Logger:
    """
    Used for logging information produced by any background service.
    """
    return logging.getLogger(BACKGROUND_LOGGER_NAME)


# This is only used by some tests.
# A better implementation is needed when the logger gets refactored.
def set_background_service_logger(log_level):
    logger = logging.getLogger(BACKGROUND_LOGGER_NAME)
    logger.setLevel(log_level)

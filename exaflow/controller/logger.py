import logging

from exaflow.controller import BACKGROUND_LOGGER_NAME
from exaflow.controller import config as ctrl_config


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
    formatter = logging.Formatter(
        f"%(asctime)s - %(levelname)s - %(module)s.%(funcName)s(%(lineno)d) - [{ctrl_config.federation}] - [exaflow-controller] - [{ctrl_config.node_identifier}] - [{request_id}] - %(message)s"
    )

    # StreamHandler
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    if log_level:
        logger.setLevel(log_level)
    else:
        logger.setLevel(ctrl_config.log_level)

    return logger


def get_background_service_logger() -> logging.Logger:
    """
    Used for logging information produced by any background service.
    """
    return logging.getLogger(BACKGROUND_LOGGER_NAME)


# this is only used by some tests.
# a better implementation needed when logger gets refactored
def set_background_service_logger(log_level):
    logger = logging.getLogger(BACKGROUND_LOGGER_NAME)
    logger.setLevel(log_level)

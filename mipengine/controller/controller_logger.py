import logging
from mipengine.controller import config as ctrl_config


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


def init_logger(request_id):
    logger = logging.getLogger(request_id)

    formatter = logging.Formatter(
        f"%(asctime)s - %(levelname)s - CONTROLLER - %(module)s - %(funcName)s(%(lineno)d) - {request_id} - %(message)s"
    )

    # StreamHandler
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.setLevel(ctrl_config.log_level)

    return logger


def get_background_service_logger():
    """
    Used for logging information produced by any background service.
    """
    return logging.getLogger("controller_background_service")

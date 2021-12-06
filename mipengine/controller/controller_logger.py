import logging
from mipengine.controller import config as ctrl_config


def getRequestLogger(context_id):
    """
    Used for logging information produced after an endpoint request.
    """
    logger = (
        initLogger(context_id)
        if not logging.getLogger(context_id).hasHandlers()
        else logging.getLogger(context_id)
    )

    return logger


def initLogger(context_id):
    logger = logging.getLogger(context_id)
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        f"%(asctime)s - %(levelname)s - CONTROLLER - context_id: {context_id} - %(message)s"
    )

    # StreamHandler
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.setLevel(ctrl_config.log_level)

    return logger


def getBackgroundServiceLogger():
    """
    Used for logging information produced by any background service.
    """
    return logging.getLogger("controller_background_service")

import logging
from mipengine.controller import config as ctrl_config


def getRequestLogger(ctx_id=None):
    """
    Used for logging information produced after an endpoint request.
    """
    logger = (
        initLogger(ctx_id)
        if not logging.getLogger(ctx_id).hasHandlers()
        else logging.getLogger(ctx_id)
    )

    return logger


def initLogger(ctx_id):
    logger = logging.getLogger(ctx_id)
    if ctx_id == None:
        formatter = logging.Formatter(f"%(message)s")
    else:
        # the name in the formatter is the name of the logger i.e. ctx_id
        formatter = logging.Formatter(
            f"%(asctime)s - %(levelname)s - CONTROLLER - ctx_id: %(name)s - %(message)s"
        )
    # StreamHandler
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    for hdlr in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(hdlr)
    logger.addHandler(sh)
    logger.setLevel(ctrl_config.log_level)

    return logger


def getBackgroundServiceLogger():
    """
    Used for logging information produced by any background service.
    """
    return logging.getLogger("controller_background_service")

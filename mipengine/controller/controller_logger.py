import logging


def getRequestLogger():
    """
    Used for logging information produced after an endpoint request.
    """
    return logging.getLogger("controller_request")


def getBackgroundServiceLogger():
    """
    Used for logging information produced by any background service.
    """
    return logging.getLogger("controller_background_service")

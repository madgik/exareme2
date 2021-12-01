import logging

from mipengine.controller import controller_logger as ctrl_logger


def test_adds_ctx_id():
    logger = ctrl_logger.getRequestLogger("123")
    logger.error("Hello there! I am the controller logger!")
    # root controller
    logging.warning("Ha HA!")

import logging
import time
from functools import wraps


def getAppLogger():
    return logging.getLogger("quart.app")


def getServerLogger():
    return logging.getLogger("quart.server")

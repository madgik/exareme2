import logging
from logging.config import dictConfig

from quart import Quart
from quart.logging import serving_handler

from mipengine.controller import config as ctrl_config
from mipengine.controller.api.algorithms_endpoint import algorithms
from mipengine.controller.api.error_handlers import error_handlers

app = Quart(__name__)
app.register_blueprint(algorithms)
app.register_blueprint(error_handlers)

dictConfig(
    {
        "version": 1,
        "formatters": {
            "app": {
                "format": "%(asctime)s - %(levelname)s - CONTROLLER - %(module)s - %(funcName)s(%(lineno)d) - %(message)s",
            },
            "server": {
                "format": "%(asctime)s - %(levelname)s - CONTROLLER - BACKGROUND - %(module)s - %(funcName)s(%(lineno)d) - %(message)s",
            },
        },
        "handlers": {
            "app": {
                "level": ctrl_config.log_level,
                "formatter": "app",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
            "server": {
                "level": ctrl_config.log_level,
                "formatter": "server",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "quart.app": {
                "level": ctrl_config.log_level,
                "handlers": ["app"],
            },
            "quart.server": {
                "level": ctrl_config.log_level,
                "handlers": ["server"],
            },
        },
    }
)

serving_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - CONTROLLER - QUART - %(message)s")
)

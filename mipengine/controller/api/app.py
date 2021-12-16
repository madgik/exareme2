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


# The initialization of the loggers is inside the app file because we don't want to initialize them
# from the tests since there is no config available then.
# We only want to intialize the loggers when the quart app is started.
dictConfig(
    {
        "version": 1,
        "formatters": {
            "controller_background_service_frm": {
                "format": "%(asctime)s - %(levelname)s - CONTROLLER - BACKGROUND - %(module)s - %(funcName)s(%(lineno)d) - %(message)s",
            },
        },
        "handlers": {
            "controller_background_service_hdl": {
                "level": ctrl_config.log_level,
                "formatter": "controller_background_service_frm",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "controller_background_service": {
                "level": ctrl_config.log_level,
                "handlers": ["controller_background_service_hdl"],
            },
            "quart.serving": {
                "level": ctrl_config.framework_log_level,
            },
        },
    }
)

serving_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - CONTROLLER - WEBAPI - %(message)s")
)

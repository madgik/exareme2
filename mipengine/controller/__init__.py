import os
from enum import Enum
from enum import unique
from importlib.resources import open_text
from logging.config import dictConfig

import envtoml

from mipengine import AttrDict
from mipengine import controller


@unique
class DeploymentType(str, Enum):
    LOCAL = "LOCAL"
    KUBERNETES = "KUBERNETES"


if config_file := os.getenv("MIPENGINE_CONTROLLER_CONFIG_FILE"):
    with open(config_file) as fp:
        config = AttrDict(envtoml.load(fp))
else:
    with open_text(controller, "config.toml") as fp:
        config = AttrDict(envtoml.load(fp))

dictConfig(
    {
        "version": 1,
        "formatters": {
            "controller_context_frm": {
                "format": "%(asctime)s - %(levelname)s - CONTROLLER - %(module)s - %(funcName)s(%(lineno)d) - %(message)s",
            },
            "controller_background_service_frm": {
                "format": "%(asctime)s - %(levelname)s - CONTROLLER - BACKGROUND - %(module)s - %(funcName)s(%(lineno)d) - %(message)s",
            },
        },
        "handlers": {
            "controller_context_hdl": {
                "level": config.log_level,
                "formatter": "controller_context_frm",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
            "controller_background_service_hdl": {
                "level": config.log_level,
                "formatter": "controller_background_service_frm",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "controller_context": {
                "level": config.log_level,
                "handlers": ["controller_context_hdl"],
            },
            "controller_background_service": {
                "level": config.log_level,
                "handlers": ["controller_background_service_hdl"],
            },
        },
    }
)

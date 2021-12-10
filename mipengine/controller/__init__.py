import logging
import os
from enum import Enum
from enum import unique
from importlib.resources import open_text
from logging.config import dictConfig

import envtoml
from quart.logging import serving_handler

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

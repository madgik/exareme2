import os
from enum import Enum
from enum import unique
from importlib.resources import open_text

import envtoml

from mipengine import AttrDict
from mipengine import controller

BACKGROUND_LOGGER_NAME = "controller_background_service"


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

import os
from enum import Enum
from enum import unique
from importlib.resources import open_text

import envtoml

from exaflow import AttrDict
from exaflow import controller

BACKGROUND_LOGGER_NAME = "controller_background_service"


@unique
class DeploymentType(str, Enum):
    LOCAL = "LOCAL"
    KUBERNETES = "KUBERNETES"


# Initializing the configurations from the config file
config = None
if config_file := os.getenv("EXAREME2_CONTROLLER_CONFIG_FILE"):
    with open(config_file) as fp:
        config = AttrDict(envtoml.load(fp))
else:
    with open_text(controller, "config.toml") as fp:
        config = AttrDict(envtoml.load(fp))

worker_landscape_aggregator = None

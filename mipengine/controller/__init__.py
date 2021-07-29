import os

import envtoml
from importlib.resources import open_text

from mipengine import controller
from mipengine import AttrDict

if config_file := os.getenv("MIPENGINE_NODE_CONTROLLER_FILE"):
    with open(config_file) as fp:
        config = AttrDict(envtoml.load(fp))
else:
    with open_text(controller, "config.toml") as fp:
        config = AttrDict(envtoml.load(fp))

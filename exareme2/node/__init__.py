import os
from importlib.resources import open_text

import envtoml

from exareme2 import AttrDict
from exareme2 import node

if config_file := os.getenv("EXAREME2_NODE_CONFIG_FILE"):
    with open(config_file) as fp:
        config = AttrDict(envtoml.load(fp))
else:
    with open_text(node, "config.toml") as fp:
        config = AttrDict(envtoml.load(fp))

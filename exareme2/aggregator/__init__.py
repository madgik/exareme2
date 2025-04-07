import os
from importlib.resources import open_text

import envtoml

from exareme2 import AttrDict
from exareme2 import aggregator

if config_file := os.getenv("EXAREME2_AGGREGATOR_CONFIG_FILE"):
    with open(config_file) as fp:
        config = AttrDict(envtoml.load(fp))
else:
    with open_text(aggregator, "config.toml") as fp:
        config = AttrDict(envtoml.load(fp))

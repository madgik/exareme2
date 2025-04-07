import os
from importlib.resources import open_text

import envtoml

from exareme2 import AttrDict
from exareme2 import worker

if config_file := os.getenv("EXAREME2_WORKER_CONFIG_FILE"):
    with open(config_file) as fp:
        config = AttrDict(envtoml.load(fp))
        config.sqlite = AttrDict({})
        config.sqlite.db_name = config.identifier
else:
    with open_text(worker, "config.toml") as fp:
        config = AttrDict(envtoml.load(fp))

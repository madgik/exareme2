import os
from importlib.resources import open_text

import envtoml
from celery import signals

from mipengine import AttrDict
from mipengine import node

DATA_TABLE_PRIMARY_KEY = "row_id"


@signals.setup_logging.connect
def setup_celery_logging(**kwargs):
    pass


if config_file := os.getenv("MIPENGINE_NODE_CONFIG_FILE"):
    with open(config_file) as fp:
        config = AttrDict(envtoml.load(fp))
else:
    with open_text(node, "config.toml") as fp:
        config = AttrDict(envtoml.load(fp))

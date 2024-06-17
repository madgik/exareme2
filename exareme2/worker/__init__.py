import os
from importlib.resources import open_text
from pathlib import Path

import envtoml

from exareme2 import AttrDict
from exareme2 import worker

PROJECT_ROOT = Path(__file__).parent.parent.parent
TEST_DATA_FOLDER = PROJECT_ROOT / "tests" / "test_data"

if config_file := os.getenv("EXAREME2_WORKER_CONFIG_FILE"):
    with open(config_file) as fp:
        config = AttrDict(envtoml.load(fp))
        config.data_path = TEST_DATA_FOLDER
else:
    with open_text(worker, "config.toml") as fp:
        config = AttrDict(envtoml.load(fp))

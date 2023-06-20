import os
from enum import Enum
from enum import unique
from importlib.resources import open_text
from typing import Dict

import envtoml

from mipengine import AttrDict
from mipengine import algorithm_classes
from mipengine import controller
from mipengine.algorithm_specification import AlgorithmSpecification
from mipengine.algorithms.longitudinal_transformer import LongitudinalTransformerRunner

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


def _get_algorithms_specifications() -> Dict[str, AlgorithmSpecification]:

    specs = {
        algo_name: algorithm.get_specification()
        for algo_name, algorithm in algorithm_classes.items()
        if algorithm.get_specification().enabled
    }
    return specs


algorithms_specifications = _get_algorithms_specifications()
transformer_specs = {
    LongitudinalTransformerRunner.get_transformer_name(): LongitudinalTransformerRunner.get_specification()
}

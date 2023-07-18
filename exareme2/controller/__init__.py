import os
from enum import Enum
from enum import unique
from importlib.resources import open_text
from typing import Dict

import envtoml

from exareme2 import AttrDict
from exareme2 import algorithm_classes
from exareme2 import controller
from exareme2.algorithms.longitudinal_transformer import LongitudinalTransformerRunner
from exareme2.algorithms.specifications import AlgorithmSpecification

BACKGROUND_LOGGER_NAME = "controller_background_service"


@unique
class DeploymentType(str, Enum):
    LOCAL = "LOCAL"
    KUBERNETES = "KUBERNETES"


if config_file := os.getenv("EXAREME2_CONTROLLER_CONFIG_FILE"):
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
transformers_specifications = {
    LongitudinalTransformerRunner.get_transformer_name(): LongitudinalTransformerRunner.get_specification()
}

import os
from enum import Enum
from enum import unique
from importlib.resources import open_text
from typing import Dict
from typing import List
from typing import Type

import envtoml

from exareme2 import Algorithm
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


def _get_algorithms_specifications(
    algorithms: List[Type[Algorithm]],
) -> Dict[str, AlgorithmSpecification]:
    specs = {}
    for algorithm in algorithms:
        if algorithm.get_specification().enabled:
            algo_name = algorithm.get_specification().name
            if algo_name in specs.keys():
                raise ValueError(
                    f"The algorithm name '{algo_name}' exists more than once in the algorithm specifications."
                )
            specs[algo_name] = algorithm.get_specification()
    return specs


algorithms_specifications = _get_algorithms_specifications(algorithm_classes.values())
transformers_specifications = {
    LongitudinalTransformerRunner.get_transformer_name(): LongitudinalTransformerRunner.get_specification()
}

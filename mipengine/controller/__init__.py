import os
from enum import Enum
from enum import unique
from importlib.resources import open_text
from pathlib import Path

import envtoml

from mipengine import ALGORITHM_FOLDERS
from mipengine import AttrDict
from mipengine import controller
from mipengine.algorithm_specification import AlgorithmSpecification

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


def _get_algorithms_specifications():
    all_algorithms = {}
    for algorithms_path in ALGORITHM_FOLDERS.split(","):
        algorithms_path = Path(algorithms_path)
        for algorithm_property_path in algorithms_path.glob("*.json"):
            try:
                with open(algorithm_property_path) as algorithm_specifications_file:
                    algorithm = AlgorithmSpecification.parse_raw(
                        algorithm_specifications_file.read()
                    )
            except Exception as e:
                raise e
            all_algorithms[algorithm.name] = algorithm

    return {
        algorithm.name: algorithm
        for algorithm in all_algorithms.values()
        if algorithm.enabled
    }


algorithms_specifications = _get_algorithms_specifications()

# algorithms_specifications: Dict[str, AlgorithmSpecification] = {
#     algorithm: algorithm.get_specification()
#     for algo_name, algorithm in algorithm_classes.get_specification()
#     if algorithm.enabled
# }

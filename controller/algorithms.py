import logging
import os
import typing
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

from dataclasses_json import dataclass_json

from controller.utils import Singleton

# TODO How can we read all algorithm.json files without relative paths?
RELATIVE_ALGORITHMS_PATH = "../../algorithms"
CROSSVALIDATION_ALGORITHM_NAME = "crossvalidation"


@dataclass_json
@dataclass
class InputData:
    label: str
    desc: str
    types: List[str]
    stattypes: List[str]
    notblank: bool
    multiple: bool
    enumslen: Optional[int] = None

    def __post_init__(self):
        allowed_types = ["real", "int", "text", "boolean"]
        if not all(elem in allowed_types for elem in self.types):
            raise ValueError(f"Input data types can include: {allowed_types}")

        allowed_stattypes = ["numerical", "nominal"]
        if not all(elem in allowed_stattypes for elem in self.stattypes):
            raise ValueError(f"Input data stattypes can include: {allowed_stattypes}")


@dataclass_json
@dataclass
class GenericParameter:
    label: str
    desc: str
    type: str
    notblank: bool
    multiple: bool
    default: 'typing.Any'
    enums: Optional[List[Any]] = None
    min: Optional[int] = None
    max: Optional[int] = None

    def __post_init__(self):
        allowed_types = ["real", "int", "text", "boolean"]
        if self.type not in allowed_types:
            raise ValueError(f"Generic parameter type can be one of the following: {allowed_types}")


@dataclass_json
@dataclass
class Algorithm:
    name: str
    desc: str
    label: str
    enabled: bool
    inputdata: Optional[Dict[str, InputData]] = None
    parameters: Optional[Dict[str, GenericParameter]] = None
    flags: Optional[Dict[str, bool]] = None


class Algorithms(metaclass=Singleton):
    crossvalidation: Algorithm
    available: Dict[str, Algorithm]

    def __init__(self):
        algorithm_property_paths = [os.path.join(RELATIVE_ALGORITHMS_PATH, json_file)
                                    for json_file in os.listdir(RELATIVE_ALGORITHMS_PATH)
                                    if json_file.endswith('.json')]

        all_algorithms = {}
        for algorithm_property_path in algorithm_property_paths:
            try:
                algorithm = Algorithm.from_json(open(algorithm_property_path).read())
            except Exception as e:
                logging.error(f"Parsing property file: {algorithm_property_path}")
                raise e
            all_algorithms[algorithm.name] = algorithm

        self.available = {
            algorithm.name: algorithm
            for algorithm in all_algorithms.values()
            if algorithm.enabled and algorithm.name != CROSSVALIDATION_ALGORITHM_NAME}

        self.crossvalidation = None
        if (CROSSVALIDATION_ALGORITHM_NAME in all_algorithms.keys() and
                all_algorithms[CROSSVALIDATION_ALGORITHM_NAME].enabled):
            self.crossvalidation = all_algorithms[CROSSVALIDATION_ALGORITHM_NAME]


Algorithms()

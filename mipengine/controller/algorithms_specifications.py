import importlib
import logging
import typing
from dataclasses import dataclass
from enum import Enum
from enum import unique
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from dataclasses_json import dataclass_json

from mipengine import ALGORITHM_FOLDERS

# TODO Enums are not supported from the dataclass_json library
# For now some helper methods are added.
# All of the helper methods and the __post_init methods should be removed with pydantic.


@unique
class InputDataStatType(str, Enum):
    NUMERICAL = "numerical"
    NOMINAL = "nominal"

    @classmethod
    def has_value(cls, item):
        return item in [v.value for v in cls.__members__.values()]

    @classmethod
    def get_values(cls):
        return [v.value for v in cls.__members__.values()]


@unique
class InputDataType(str, Enum):
    REAL = "real"
    INT = "int"
    TEXT = "text"

    @classmethod
    def has_value(cls, item):
        return item in [v.value for v in cls.__members__.values()]

    @classmethod
    def get_values(cls):
        return [v.value for v in cls.__members__.values()]


@dataclass_json
@dataclass
class InputDataSpecification:
    label: str
    desc: str
    types: List[str]
    stattypes: List[str]
    notblank: bool
    multiple: bool
    enumslen: Optional[int] = None

    def __post_init__(self):
        if not all(InputDataType.has_value(elem) for elem in self.types):
            raise ValueError(
                f"Input data types can include: {InputDataType.get_values()}"
            )

        if not all(InputDataStatType.has_value(elem) for elem in self.stattypes):
            raise ValueError(
                f"Input data stattypes can include: {InputDataStatType.get_values()}"
            )


@dataclass_json
@dataclass
class InputDataSpecifications:
    x: Optional[InputDataSpecification] = None
    y: Optional[InputDataSpecification] = None


@unique
class ParameterType(str, Enum):
    REAL = "real"
    INT = "int"
    TEXT = "text"
    BOOLEAN = "boolean"

    @classmethod
    def has_value(cls, item):
        return item in [v.value for v in cls.__members__.values()]

    @classmethod
    def get_values(cls):
        return [v.value for v in cls.__members__.values()]


@dataclass_json
@dataclass
class ParameterSpecification:
    label: str
    desc: str
    type: str
    notblank: bool
    multiple: bool
    default: "typing.Any"
    enums: Optional[List[Any]] = None
    min: Optional[int] = None
    max: Optional[int] = None


@dataclass_json
@dataclass
class AlgorithmSpecifications:
    name: str
    desc: str
    label: str
    enabled: bool
    inputdata: Optional[InputDataSpecifications] = None
    parameters: Optional[Dict[str, ParameterSpecification]] = None
    flags: Optional[Dict[str, bool]] = None


class AlgorithmsSpecifications:
    enabled_algorithms: Dict[str, AlgorithmSpecifications]

    def __init__(self):
        all_algorithms = {}
        for algorithms_path in ALGORITHM_FOLDERS.split(","):
            algorithms_path = Path(algorithms_path)
            for algorithm_property_path in algorithms_path.glob("*.json"):
                try:
                    algorithm = AlgorithmSpecifications.from_json(
                        open(algorithm_property_path).read()
                    )
                except Exception as e:
                    logging.error(f"Parsing property file: {algorithm_property_path}")
                    raise e
                all_algorithms[algorithm.name] = algorithm

            # The algorithm key should be in snake case format, to make searching for an algorithm easier.
            self.enabled_algorithms = {
                algorithm.name: algorithm
                for algorithm in all_algorithms.values()
                if algorithm.enabled
            }


algorithms_specifications = AlgorithmsSpecifications()

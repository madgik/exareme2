import logging
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from dataclasses_json import dataclass_json

from mipengine import algorithms


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
        allowed_types = ["real", "int", "text", "boolean"]
        if not all(elem in allowed_types for elem in self.types):
            raise ValueError(f"Input data types can include: {allowed_types}")

        allowed_stattypes = ["numerical", "nominal"]
        if not all(elem in allowed_stattypes for elem in self.stattypes):
            raise ValueError(f"Input data stattypes can include: {allowed_stattypes}")


@dataclass_json
@dataclass
class GenericParameterSpecification:
    label: str
    desc: str
    type: str
    notblank: bool
    multiple: bool
    default: "typing.Any"
    enums: Optional[List[Any]] = None
    min: Optional[int] = None
    max: Optional[int] = None

    def __post_init__(self):
        allowed_types = ["real", "int", "text", "boolean"]
        if self.type not in allowed_types:
            raise ValueError(
                f"Generic parameter type can be one of the following: {allowed_types}"
            )


@dataclass_json
@dataclass
class AlgorithmSpecifications:
    name: str
    desc: str
    label: str
    enabled: bool
    inputdata: Optional[Dict[str, InputDataSpecification]] = None
    parameters: Optional[Dict[str, GenericParameterSpecification]] = None
    flags: Optional[Dict[str, bool]] = None


class AlgorithmsSpecifications:
    enabled_algorithms: Dict[str, AlgorithmSpecifications]

    def __init__(self):
        algorithms_path = Path(algorithms.__file__).parent

        all_algorithms = {}
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

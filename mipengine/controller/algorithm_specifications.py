import logging
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from pydantic import BaseModel

from mipengine import ALGORITHM_FOLDERS
from mipengine.controller.api.algorithm_specifications_dtos import (
    AlgorithmSpecificationDTO,
)
from mipengine.controller.api.algorithm_specifications_dtos import (
    AlgorithmSpecificationsDTO,
)
from mipengine.controller.api.algorithm_specifications_dtos import (
    InputDataSpecificationDTO,
)
from mipengine.controller.api.algorithm_specifications_dtos import (
    InputDataSpecificationsDTO,
)
from mipengine.controller.api.algorithm_specifications_dtos import InputDataStatType
from mipengine.controller.api.algorithm_specifications_dtos import InputDataType
from mipengine.controller.api.algorithm_specifications_dtos import (
    ParameterEnumSpecificationDTO,
)
from mipengine.controller.api.algorithm_specifications_dtos import ParameterEnumType
from mipengine.controller.api.algorithm_specifications_dtos import (
    ParameterSpecificationDTO,
)
from mipengine.controller.api.algorithm_specifications_dtos import ParameterType


class InputDataSpecification(BaseModel):
    label: str
    desc: str
    types: List[InputDataType]
    stattypes: List[InputDataStatType]
    notblank: bool
    multiple: bool
    enumslen: Optional[int]

    def convert_to_inputdata_specification_dto(self):
        # The only difference of the DTO is that it's stattypes is Optional,
        # due to the fact that the datasets/data_model variables are added.
        return InputDataSpecificationDTO(
            label=self.label,
            desc=self.desc,
            types=self.types,
            stattypes=self.stattypes,
            notblank=self.notblank,
            multiple=self.multiple,
            enumslen=self.enumslen,
        )


def get_data_model_input_data_specification_DTO():
    return InputDataSpecificationDTO(
        label="Data model of the data.",
        desc="The data model that the algorithm will run on.",
        types=[InputDataType.TEXT],
        notblank=True,
        multiple=False,
        stattypes=None,
        enumslen=None,
    )


def get_datasets_input_data_specification_DTO():
    return InputDataSpecificationDTO(
        label="Set of data to use.",
        desc="The set of data to run the algorithm on.",
        types=[InputDataType.TEXT],
        notblank=True,
        multiple=True,
        stattypes=None,
        enumslen=None,
    )


def get_filters_input_data_specification_DTO():
    return InputDataSpecificationDTO(
        label="filter on the data.",
        desc="Features used in my algorithm.",
        types=[InputDataType.JSONOBJECT],
        notblank=False,
        multiple=False,
        stattypes=None,
        enumslen=None,
    )


class InputDataSpecifications(BaseModel):
    y: InputDataSpecification
    x: Optional[InputDataSpecification]

    def convert_to_inputdata_specifications_dto(self):
        # In the DTO the datasets, data_model and filter parameters are added from the engine.
        # These parameters are not added by the algorithm developer.
        y = self.y.convert_to_inputdata_specification_dto()
        x = self.x.convert_to_inputdata_specification_dto() if self.x else None
        return InputDataSpecificationsDTO(
            y=y,
            x=x,
            data_model=get_data_model_input_data_specification_DTO(),
            datasets=get_datasets_input_data_specification_DTO(),
            filter=get_filters_input_data_specification_DTO(),
        )


class ParameterEnumSpecification(BaseModel):
    type: ParameterEnumType
    source: Any

    def convert_to_parameter_enum_specification_dto(self):
        return ParameterEnumSpecificationDTO(
            type=self.type,
            source=self.source,
        )


class ParameterSpecification(BaseModel):
    label: str
    desc: str
    types: List[ParameterType]
    notblank: bool
    multiple: bool
    default: Any
    enums: Optional[ParameterEnumSpecification]
    min: Optional[float]
    max: Optional[float]

    def convert_to_parameter_specification_dto(self):
        return ParameterSpecificationDTO(
            label=self.label,
            desc=self.desc,
            types=self.types,
            notblank=self.notblank,
            multiple=self.multiple,
            default=self.default,
            enums=self.enums.convert_to_parameter_enum_specification_dto()
            if self.enums
            else None,
            min=self.min,
            max=self.max,
        )


class AlgorithmSpecification(BaseModel):
    name: str
    desc: str
    label: str
    enabled: bool
    inputdata: InputDataSpecifications
    parameters: Optional[Dict[str, ParameterSpecification]]
    flags: Optional[Dict[str, bool]]

    def convert_to_algorithm_specifications_dto(self):
        # Converting to a DTO does not include the flags.
        return AlgorithmSpecificationDTO(
            name=self.name,
            desc=self.desc,
            label=self.label,
            inputdata=self.inputdata.convert_to_inputdata_specifications_dto(),
            parameters={
                name: value.convert_to_parameter_specification_dto()
                for name, value in self.parameters.items()
            }
            if self.parameters
            else None,
        )


class AlgorithmSpecifications:
    enabled_algorithms: Dict[str, AlgorithmSpecification]

    def __init__(self):
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
                    logging.error(f"Parsing property file: {algorithm_property_path}")
                    raise e
                all_algorithms[algorithm.name] = algorithm

            # The algorithm key should be in snake case format, to make searching for an algorithm easier.
            self.enabled_algorithms = {
                algorithm.name: algorithm
                for algorithm in all_algorithms.values()
                if algorithm.enabled
            }

    def get_enabled_algorithm_dtos(self) -> AlgorithmSpecificationsDTO:
        return AlgorithmSpecificationsDTO(
            __root__=[
                algorithm.convert_to_algorithm_specifications_dto()
                for algorithm in self.enabled_algorithms.values()
            ]
        )


algorithm_specifications = AlgorithmSpecifications()

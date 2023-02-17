from abc import ABC
from enum import Enum
from enum import unique
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from pydantic import BaseModel


class ImmutableBaseModel(BaseModel, ABC):
    class Config:
        allow_mutation = False


@unique
class InputDataType(Enum):
    REAL = "real"
    INT = "int"
    TEXT = "text"
    JSONOBJECT = "jsonObject"


@unique
class InputDataStatType(Enum):
    NUMERICAL = "numerical"
    NOMINAL = "nominal"


@unique
class ParameterType(str, Enum):
    REAL = "real"
    INT = "int"
    TEXT = "text"
    BOOLEAN = "boolean"


@unique
class ParameterEnumType(str, Enum):
    LIST = "list"
    INPUT_VAR_CDE_ENUMS = "input_var_CDE_enums"
    FIXED_VAR_CDE_ENUMS = "fixed_var_CDE_enums"
    INPUT_VAR_NAMES = "input_var_names"


class InputDataSpecificationDTO(ImmutableBaseModel):
    label: str
    desc: str
    types: List[InputDataType]
    notblank: bool
    multiple: bool
    stattypes: Optional[List[InputDataStatType]]
    enumslen: Optional[int]


class InputDataSpecificationsDTO(ImmutableBaseModel):
    data_model: InputDataSpecificationDTO
    datasets: InputDataSpecificationDTO
    filter: InputDataSpecificationDTO
    y: InputDataSpecificationDTO
    x: Optional[InputDataSpecificationDTO]


class ParameterEnumSpecificationDTO(ImmutableBaseModel):
    type: ParameterEnumType
    source: Any


class ParameterSpecificationDTO(ImmutableBaseModel):
    label: str
    desc: str
    types: List[ParameterType]
    notblank: bool
    multiple: bool
    default: Any
    enums: Optional[ParameterEnumSpecificationDTO]
    min: Optional[float]
    max: Optional[float]


class AlgorithmSpecificationDTO(ImmutableBaseModel):
    name: str
    desc: str
    label: str
    inputdata: InputDataSpecificationsDTO
    parameters: Optional[Dict[str, ParameterSpecificationDTO]]


class AlgorithmSpecificationsDTO(ImmutableBaseModel):
    __root__: List[AlgorithmSpecificationDTO]

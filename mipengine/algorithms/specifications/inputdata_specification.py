from enum import Enum
from enum import unique
from typing import List
from typing import Optional

from mipengine.algorithms.specifications import ImmutableBaseModel


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


class InputDataSpecification(ImmutableBaseModel):
    label: str
    desc: str
    types: List[InputDataType]
    stattypes: List[InputDataStatType]
    notblank: bool
    multiple: bool
    enumslen: Optional[int]


class InputDataSpecifications(ImmutableBaseModel):
    y: InputDataSpecification
    x: Optional[InputDataSpecification]

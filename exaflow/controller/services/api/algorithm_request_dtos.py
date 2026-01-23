from abc import ABC
from enum import Enum
from enum import unique
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from pydantic import BaseModel


@unique
class AlgorithmRequestSystemFlags(str, Enum):
    SMPC = "smpc"


class ImmutableBaseModel(BaseModel, ABC):
    class Config:
        allow_mutation = False


class AlgorithmInputDataDTO(ImmutableBaseModel):
    data_model: str
    datasets: List[str]
    validation_datasets: Optional[List[str]]
    filters: Optional[dict]
    y: Optional[List[str]]
    x: Optional[List[str]]


PARAMETERS_TYPE = Dict[str, Any]


class AlgorithmRequestDTO(BaseModel):
    request_id: Optional[str]
    inputdata: AlgorithmInputDataDTO
    parameters: Optional[PARAMETERS_TYPE]
    flags: Optional[Dict[str, Any]]
    preprocessing: Optional[Dict[str, PARAMETERS_TYPE]]

from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from pydantic import BaseModel

USE_SMPC_FLAG = "smpc"


class AlgorithmInputDataDTO(BaseModel):
    data_model: str
    datasets: List[str]
    filters: dict = None
    y: Optional[List[str]]
    x: Optional[List[str]]


class AlgorithmRequestDTO(BaseModel):
    request_id: Optional[str] = None
    inputdata: AlgorithmInputDataDTO
    parameters: Optional[Dict[str, Any]] = None
    flags: Optional[Dict[str, Any]] = None

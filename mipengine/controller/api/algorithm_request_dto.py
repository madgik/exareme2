from pydantic import BaseModel
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

USE_SMPC_FLAG = "smpc"


class AlgorithmInputDataDTO(BaseModel):
    pathology: str
    datasets: List[str]
    filters: dict = None
    x: Optional[List[str]] = None
    y: Optional[List[str]] = None


class AlgorithmRequestDTO(BaseModel):
    request_id: Optional[str] = None
    inputdata: AlgorithmInputDataDTO
    parameters: Optional[Dict[str, Any]] = None
    flags: Optional[Dict[str, Any]] = None


# @dataclass_json
# @dataclass
# class VariablesSet:
# 	name: str
# 	column_names: List[str]

# @dataclass_json
# @dataclass
# class AlgorithmRequest:
#     #algorithm_name: str
#     pathology: str
#     datasets: List[str]
#     columns: List[VariablesSet]
#     filter: Optional[Any] = None
#     parameters: Optional[Dict[str, Any]] = None

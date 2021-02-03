from dataclasses import dataclass
from typing import Dict, Any, Optional, List

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class AlgorithmInputDataDTO:
    pathology: str
    dataset: List[str]
    filter: Optional[Any] = None
    x: Optional[List[str]] = None
    y: Optional[List[str]] = None


@dataclass_json
@dataclass
class AlgorithmRequestDTO:
    inputdata: AlgorithmInputDataDTO
    parameters: Optional[Dict[str, Any]] = None
    crossvalidation: Optional[Dict[str, Any]] = None

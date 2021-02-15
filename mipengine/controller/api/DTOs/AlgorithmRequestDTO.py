from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class AlgorithmInputDataDTO:
    pathology: str
    datasets: List[str]
    filters: Optional[Any] = None
    x: Optional[List[str]] = field(default_factory=list)
    y: Optional[List[str]] = field(default_factory=list)


@dataclass_json
@dataclass
class AlgorithmRequestDTO:
    inputdata: AlgorithmInputDataDTO
    parameters: Optional[Dict[str, Any]] = None
    crossvalidation: Optional[Dict[str, Any]] = None

from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import Optional

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class AlgorithmRequestDTO:
    inputdata: Dict[str, Any]
    parameters: Optional[Dict[str, Any]] = None

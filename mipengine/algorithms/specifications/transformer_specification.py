from enum import Enum
from enum import unique
from typing import Dict
from typing import List
from typing import Optional

from mipengine.algorithms.specifications.parameter_specification import (
    ParameterSpecification,
)
from mipengine.algorithms.specifications.pipeline_step_specification import (
    PipelineStepSpecification,
)


@unique
class TransformerName(Enum):
    LONGITUDINAL_TRANSFORM = "longitudinal_transform"


class TransformerSpecification(PipelineStepSpecification):
    name: str
    desc: str
    label: str
    enabled: bool
    parameters: Optional[Dict[str, ParameterSpecification]]
    compatible_algorithms: Optional[List]

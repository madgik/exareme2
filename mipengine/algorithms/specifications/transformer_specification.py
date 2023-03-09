from typing import Dict
from typing import Optional

from mipengine.algorithms.specifications.parameter_specification import (
    ParameterSpecification,
)
from mipengine.algorithms.specifications.pipeline_step_specification import (
    PipelineStepSpecification,
)


class TransformerSpecification(PipelineStepSpecification):
    name: str
    desc: str
    label: str
    enabled: bool
    parameters: Optional[Dict[str, ParameterSpecification]]

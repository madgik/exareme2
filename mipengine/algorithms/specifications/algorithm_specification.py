from typing import Dict
from typing import Optional

from mipengine.algorithms.specifications.inputdata_specification import (
    InputDataSpecifications,
)
from mipengine.algorithms.specifications.parameter_specification import (
    ParameterSpecification,
)
from mipengine.algorithms.specifications.pipeline_step_specification import (
    PipelineStepSpecification,
)


class AlgorithmSpecification(PipelineStepSpecification):
    name: str
    desc: str
    label: str
    enabled: bool
    inputdata: InputDataSpecifications
    parameters: Optional[Dict[str, ParameterSpecification]]
    flags: Optional[Dict[str, bool]]

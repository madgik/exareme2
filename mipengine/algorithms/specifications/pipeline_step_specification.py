from pydantic import root_validator

from mipengine.algorithms.specifications import ImmutableBaseModel
from mipengine.algorithms.specifications.parameter_specification import (
    validate_parameters,
)


class PipelineStepSpecification(ImmutableBaseModel):
    @root_validator
    def validate_parameters(cls, cls_values):
        validate_parameters(cls_values)
        return cls_values

from abc import ABC
from abc import abstractmethod

from mipengine.algorithms.base_classes.pipeline_step import PipelineStep
from mipengine.algorithms.specifications.transformer_specification import (
    TransformerSpecification,
)


class Transformer(PipelineStep, ABC):
    """
    This is the abstract class that all transformers must implement.
    """

    # TODO This logic cannot be inherited, should we find another approach?
    def __init_subclass__(cls, stepname, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.stepname = stepname

    @staticmethod
    @abstractmethod
    def get_specification() -> TransformerSpecification:
        pass

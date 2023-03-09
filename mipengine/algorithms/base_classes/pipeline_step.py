from abc import ABC
from abc import abstractmethod
from typing import Union

from mipengine.algorithms.specifications.algorithm_specification import (
    AlgorithmSpecification,
)
from mipengine.algorithms.specifications.transformer_specification import (
    TransformerSpecification,
)


class PipelineStep(ABC):
    """
    This is the abstract class that all pipeline step classes must implement.
    """

    @staticmethod
    @abstractmethod
    def get_specification() -> Union[AlgorithmSpecification, TransformerSpecification]:
        pass

    @abstractmethod
    def run(self, engine):
        # The executor must be available only inside run()
        # The reasoning for this is that executor.data_model_views must already be
        # available when the executor is available to the algorithm, but the creation of
        # the executor.data_model_views requires calls to
        # algorithm.get_variable_groups(), algorithm.get_check_min_rows() and get_dropna().
        """
        The implementation of the algorithm flow logic goes in this method.
        """
        pass

    # TODO More work is needed here on what methods will belong to the pipeline step vs Algorithm and Transformer classes
    # TODO https://team-1617704806227.atlassian.net/browse/MIP-755

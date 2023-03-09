from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from pydantic import BaseModel

from mipengine.algorithms.base_classes.pipeline_step import PipelineStep
from mipengine.algorithms.specifications.algorithm_specification import (
    AlgorithmSpecification,
)


class Variables(BaseModel):
    x: List[str]
    y: List[str]

    class Config:
        arbitrary_types_allowed = True


class InitializationParams(BaseModel):
    algorithm_name: str
    variables: Variables
    var_filters: Optional[dict] = None
    algorithm_parameters: Optional[Dict[str, Any]] = None
    metadata: Dict[str, dict]

    class Config:
        arbitrary_types_allowed = True


class Algorithm(PipelineStep, ABC):
    """
    This is the abstract class that all algorithm classes must implement.
    """

    # TODO This logic cannot be inherited, should we find another approach?
    def __init_subclass__(cls, stepname, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.stepname = stepname

    def __init__(self, initialization_params: InitializationParams):
        """
        Parameters
        ----------
        initialization_params : InitializationParams
        """
        self._initialization_params = initialization_params

    @property
    def variables(self) -> Variables:
        """
        Returns
        -------
        Variables
            The variables
        """
        return self._initialization_params.variables

    @property
    def algorithm_parameters(self) -> Dict[str, Any]:
        """
        Returns
        -------
        Dict[str,Any]
            The algorithm parameters
        """
        return self._initialization_params.algorithm_parameters

    @property
    def metadata(self) -> Dict[str, dict]:
        """
        Returns
        -------
        Dist[str,dict]
            The variables' metadata
        """
        return self._initialization_params.metadata

    @abstractmethod
    def get_variable_groups(self) -> List[List[str]]:
        """
        This method must be implemented to return the variable groups from which the
        data model view tables will be created. The algorithm execution engine
        will take care of creating the data model view tables on the nodes' dbs. The
        data model views can be accessed from the algorithm flow code via
        engine.data_model_views list inside the run() method .

        Returns
        -------
        List[List[str]]
            The variable groups
        """
        pass

    def get_dropna(self) -> bool:
        """
        If an algorithm needs to keep the 'Not Available' values in its data model view
        tables, this method must be overridden to return False. The algorithm execution
        infrastructure will access this value when the data model view tables on the
        nodes' dbs are created.

        Returns
        -------
        bool
        """
        return True

    def get_check_min_rows(self) -> bool:
        """
        If an algorithm needs to ignore the minimum row count threshold for its data
        model view tables, this method must be overridden to return False. The algorithm
        execution infrastructure will access this value when the data model view tables
        on the nodes' dbs are created.

        Returns
        -------
        bool
        """
        return True

    @staticmethod
    @abstractmethod
    def get_specification() -> AlgorithmSpecification:
        pass

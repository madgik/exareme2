from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from pydantic import BaseModel


class Variables(BaseModel):
    x: List[str]
    y: List[str]

    class Config:
        arbitrary_types_allowed = True
        allow_mutation = False


class AlgorithmDataLoader(ABC):
    def __init__(self, variables: Variables):
        self._variables = variables

    def __init_subclass__(cls, algname, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.algname = algname

    @abstractmethod
    def get_variable_groups(self) -> List[List[str]]:
        """
        This method must be implemented to return the variable groups from which the
        data model view tables will be created. The algorithm execution engine
        will take care of creating the data model view tables on the workers' dbs. The
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
        engine will access this value when the data model view tables on the
        workers' dbs are created.

        Returns
        -------
        bool
        """
        return True

    def get_check_min_rows(self) -> bool:
        """
        If an algorithm needs to ignore the minimum row count threshold for its data
        model view tables, this method must be overridden to return False. The algorithm
        execution engine will access this value when the data model view tables
        on the workers' dbs are created.

        Returns
        -------
        bool
        """
        return True

    def get_variables(self) -> Variables:
        return self._variables


class InitializationParams(BaseModel):
    algorithm_name: str
    var_filters: Optional[dict] = None
    algorithm_parameters: Optional[Dict[str, Any]] = None
    datasets: List[str]

    class Config:
        arbitrary_types_allowed = True
        allow_mutation = False


class Algorithm(ABC):
    """
    This is the abstract class that all algorithm flow classes must implement. The class
    can be named arbitrarily, it will be detected by its 'algname' attribute

    Attributes
    ----------
    algname : str
    """

    def __init__(
        self,
        initialization_params: InitializationParams,
        data_loader: AlgorithmDataLoader,
        engine,
    ):
        """
        Parameters
        ----------
        initialization_params : InitializationParams
        """
        self._initialization_params = initialization_params
        self._data_loader = data_loader
        self._engine = engine

    def __init_subclass__(cls, algname: str, **kwargs):
        """
        Parameters
        ----------
        algname : str
            The algorithm name, as defined in the "name" field in the <algorithm>.json
        """
        super().__init_subclass__(**kwargs)
        cls.algname = algname

    @property
    def engine(self):
        return self._engine

    @property
    def variables(self) -> Variables:
        """
        Returns
        -------
        Variables
            The variables
        """
        return self._data_loader.get_variables()

    @property
    def variable_groups(self) -> List[List[str]]:
        """
        Use this property when the variable_groups, as defined in
        AlgorithmDataLoader.get_variable_groups(), need to be accessed from the
        Algorithm class.
        """
        return self._data_loader.get_variable_groups()

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
    def datasets(self) -> List[str]:
        return self._initialization_params.datasets

    @abstractmethod
    def run(self, data, metadata: dict):
        """
        The implementation of the algorithm flow logic goes in this method.
        """
        pass

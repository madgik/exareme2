from abc import ABC
from abc import abstractmethod
from typing import List


class Algorithm(ABC):
    """
    This is the abstract class that all algorithm flow classes must implement. The class
    can be named arbitrarily, it will be detected by its 'algname' attribute

    Attributes
    ----------
    algname : str
    executor : _AlgorithmExecutionInterface
    """

    def __init__(self, executor):
        """
        Parameters
        ----------
        executor : _AlgorithmExecutionInterface
            The executor attribute gives access to the algorithm execution infrastructure.
        """
        self._executor = executor

    def __init_subclass__(cls, algname, **kwargs):
        """
        Parameters
        ----------
        algname : str
            The algorithm name, as defined in the "name" field in the <algorithm>.json
        """
        super().__init_subclass__(**kwargs)
        cls.algname = algname

    @property
    def executor(self):
        """
        Returns
        -------
        executor : _AlgorithmExecutionInterface
            The executor attribute gives access to the algorithm execution infrastructure.
        """
        return self._executor

    @abstractmethod
    def get_variable_groups(self) -> List[List[str]]:
        """
        This method must be implemented to return the variable groups from which the
        data model view tables will be created. The algorithm execution infrastructure
        will take care of creating the data model view tables on the nodes' dbs. The
        data model views can be accessed from the algorithm flow code via
        self.executor.data_model_views list.

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

    @abstractmethod
    def run(self):
        """
        The implementation of the algorithm flow logic goes in this method.
        """
        pass

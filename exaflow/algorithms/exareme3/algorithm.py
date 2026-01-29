from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import Optional

from exaflow.algorithms.utils.inputdata_utils import Inputdata


class Algorithm(ABC):
    algname: str

    def __init__(
        self,
        *,
        inputdata: Inputdata,
        engine,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        self._inputdata = inputdata
        self._engine = engine
        self._parameters = parameters if parameters is not None else {}

    def __init_subclass__(cls, algname: str, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.algname = algname

    def run_local_udf(self, func, kw_args):
        return self._engine.run_udf(
            func,
            self.drop_na_rows,
            self.check_min_rows,
            self.add_dataset_variable,
            kw_args,
        )

    @property
    def inputdata(self):
        return self._inputdata

    @property
    def parameters(self) -> Dict[str, Any]:
        return self._parameters

    @property
    def drop_na_rows(self) -> bool:
        """
        By default, the rows with 'Not Available' values are dropped.
        If an algorithm needs to keep the 'Not Available' values,
        this method must be overridden to return False.
        """
        return True

    @property
    def check_min_rows(self) -> bool:
        """
        If an algorithm needs to ignore the minimum row count threshold check,
        for its data, this method must be overridden to return False.
        """
        return True

    @property
    def add_dataset_variable(self) -> bool:
        """
        If an algorithm needs to include the dataset column, in addition to the user
        selected columns, this method should be overridden to return True.
        """
        return False

    @abstractmethod
    def run(self, metadata: dict):
        """
        Execute the algorithm flow.

        Parameters
        ----------
        metadata : dict
            Per-variable metadata. Flows rely on at least `metadata[var]["is_categorical"]`
            to decide encoding, and may use additional keys as needed.
        """
        pass

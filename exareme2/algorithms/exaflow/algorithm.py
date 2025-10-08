from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import Optional

from exareme2.algorithms.utils.inputdata_utils import Inputdata


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

    @property
    def engine(self):
        return self._engine

    @property
    def inputdata(self):
        return self._inputdata

    @property
    def parameters(self) -> Dict[str, Any]:
        return self._parameters

    @abstractmethod
    def run(self, metadata: dict):
        pass

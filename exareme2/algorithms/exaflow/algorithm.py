from abc import ABC
from abc import abstractmethod

from exareme2.algorithms.utils.inputdata_utils import Inputdata


class Algorithm(ABC):
    algname: str

    def __init__(self, *, inputdata: Inputdata, engine):
        self._inputdata = inputdata
        self._engine = engine

    def __init_subclass__(cls, algname: str, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.algname = algname

    @property
    def engine(self):
        return self._engine

    @property
    def inputdata(self):
        return self._inputdata

    @abstractmethod
    def run(self, metadata: dict):
        pass

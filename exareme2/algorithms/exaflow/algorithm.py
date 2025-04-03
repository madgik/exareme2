from abc import ABC
from abc import abstractmethod


class Algorithm(ABC):
    algname: str

    def __init__(self, *, inputdata: dict, engine):
        self._inputdata = inputdata
        self._engine = engine

    def __init_subclass__(cls, algname: str, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.algname = algname

    @property
    def engine(self):
        return self._engine

    @abstractmethod
    def run(self, inputdata: dict, metadata: dict):
        pass

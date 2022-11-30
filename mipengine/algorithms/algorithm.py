from abc import ABC
from abc import abstractmethod
from typing import List


class Algorithm(ABC):
    def __init_subclass__(cls, algname, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.algname = algname

    def __init__(self, executor):
        self._executor = executor

    @property
    def executor(self):
        return self._executor

    @abstractmethod
    def get_variable_groups(self) -> List[List[str]]:
        pass

    def get_dropna(self) -> bool:
        return True

    def get_check_min_rows(self) -> bool:
        return True

    @abstractmethod
    def run(self):
        pass

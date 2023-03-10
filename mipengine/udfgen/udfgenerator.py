from abc import ABC
from abc import abstractmethod
from typing import List

from mipengine.udfgen.udfgen_DTOs import UDFGenTableResult


class UdfGenerator(ABC):
    @abstractmethod
    def get_definition(self, udf_name: str, output_table_names: List[str]) -> str:
        pass

    @abstractmethod
    def get_exec_stmt(self, udf_name: str, output_table_names: List[str]) -> str:
        pass

    @abstractmethod
    def get_results(self, output_table_names: List[str]) -> List[UDFGenTableResult]:
        pass

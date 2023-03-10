from abc import ABC
from abc import abstractmethod
from typing import List

from mipengine.udfgen.adhoc_udfgenerator import AdhocUdfGenerator
from mipengine.udfgen.py_udfgenerator import PyUdfGenerator
from mipengine.udfgen.udfgen_DTOs import UDFGenTableResult


class UdfGenerator(ABC):
    def __new__(cls, udfregistry, func_name, **kwargs):
        if func_name in udfregistry:
            return PyUdfGenerator(udfregistry, func_name, **kwargs)
        elif AdhocUdfGenerator.is_registered(func_name):
            return AdhocUdfGenerator.get_subclass(func_name)(**kwargs)
        raise ValueError(f"UDF named '{func_name}' not found in registries.")

    @abstractmethod
    def get_definition(self, udf_name: str, output_table_names: List[str]) -> str:
        pass

    @abstractmethod
    def get_exec_stmt(self, udf_name: str, output_table_names: List[str]) -> str:
        pass

    @abstractmethod
    def get_results(self, output_table_names: List[str]) -> List[UDFGenTableResult]:
        pass


UdfGenerator.register(PyUdfGenerator)
UdfGenerator.register(AdhocUdfGenerator)

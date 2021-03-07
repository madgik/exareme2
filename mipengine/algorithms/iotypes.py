from abc import ABC, abstractmethod
from numbers import Number
from typing import Text, Tuple
from typing import TypeVar
from typing import Any
from typing import Generic
from typing import get_type_hints

import numpy as np
import pandas as pd

UDF_REGISTRY = {}

LiteralParameterT = Any
ScalarT = TypeVar("ScalarT", Number, Text, bool, np.ndarray, list, dict, tuple)


def udf(func):
    global UDF_REGISTRY
    module_name = func.__module__.split(".")[-1]
    func_name = func.__name__
    qualname = module_name + "." + func_name
    UDF_REGISTRY[qualname] = func
    return func


# -------------------------------------------------------- #
# Parametrize tensors with shape, to avoid computing their #
# output size                                              #
# -------------------------------------------------------- #
Shape = TypeVar("Shape")
Schema = TypeVar("Schema")
NRows = TypeVar("NRows")
NColumns = TypeVar("NColumns")
DType = TypeVar("DType")


class TableT(ABC):
    @abstractmethod
    def __init__(self) -> None:
        raise NotImplementedError

    def __repr__(self) -> str:
        cls = type(self).__name__
        attrs = self.__dict__
        attrs_rep = str(attrs).replace("'", "").replace(": ", "=").strip("{}")
        rep = f"{cls}({attrs_rep})"
        return rep


class RelationT(TableT, Generic[NColumns]):
    def __init__(self, ncolumns: NColumns) -> None:
        self.ncolumns = ncolumns


class LoopbackRelationT(RelationT, Generic[NColumns]):
    pass


class RelationFromSchemaT(TableT, Generic[Schema]):
    def __init__(self, schema) -> None:
        self.schema = [(col.name, col.dtype) for col in schema]

    def as_sql_parameters(self, name):
        return ",".join([f"{name}_{colname} {tp}" for colname, tp in self.schema])

    def as_sql_return_type(self, name):
        return f"TABLE({self.as_sql_parameters(name)})"


class LoopbackRelationFromSchemaT(RelationFromSchemaT, Generic[Schema]):
    pass


class TensorT(TableT, Generic[DType, NRows, NColumns]):
    def __init__(self, dtype: DType, nrows: NRows, ncolumns: NColumns) -> None:
        self.dtype = dtype
        self.nrows = nrows
        self.ncolumns = ncolumns

    @property
    def shape(self) -> tuple:
        return self.nrows, self.ncolumns


class LoopbackTensorT(TensorT, Generic[DType, NRows, NColumns]):
    pass


T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")
R1 = TypeVar("R1")
R2 = TypeVar("R2")
C1 = TypeVar("C1")
C2 = TypeVar("C2")


def f(t1: TensorT[T1, R1, C1], t2: TensorT[T2, R2, C2]) -> TensorT[T1, R1, C2]:
    return t1


R = TypeVar("R")
C = TypeVar("C")


def g(t: RelationT[C]) -> TensorT(int, R, 3):  # type: ignore
    pass

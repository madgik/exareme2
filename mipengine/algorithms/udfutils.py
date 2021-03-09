from abc import ABC, abstractmethod
from inspect import signature
from numbers import Number
from typing import Text
from typing import Tuple
from typing import get_origin
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
    if qualname in UDF_REGISTRY:
        raise KeyError(f"A UDF named {qualname} already in UDF_REGISTRY.")
    validate_type_hints(func)
    UDF_REGISTRY[qualname] = func
    return func


def validate_type_hints(func):
    allowed_types = {
        RelationT,
        LoopbackRelationT,
        TensorT,
        LoopbackTensorT,
        LiteralParameterT,
        ScalarT,
    }
    sig = signature(func)
    func_type_hints = set(
        get_origin(param.annotation)
        if get_origin(param.annotation)
        else param.annotation
        for param in sig.parameters.values()
    )
    func_type_hints.add(
        get_origin(sig.return_annotation)
        if get_origin(sig.return_annotation)
        else sig.return_annotation
    )
    if func_type_hints - allowed_types:
        raise TypeError("Function parameters are not properly annotated.")


DType = TypeVar("DType")
Schema = TypeVar("Schema")
NDims = TypeVar("NDims")


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


class RelationT(TableT, Generic[Schema]):
    def __init__(self, schema) -> None:
        self.schema = [(col.name, col.dtype) for col in schema]


class LoopbackRelationT(RelationT, Generic[Schema]):
    pass


class TensorT(TableT, Generic[DType, NDims]):
    is_generic = True

    def __init__(self, dtype, ndims) -> None:
        self.is_generic = False
        self.dtype = dtype
        self.ndims = ndims


class LoopbackTensorT(TensorT, Generic[DType, NDims]):
    pass

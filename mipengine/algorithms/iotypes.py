from typing import TypeVar
from typing import Any

import numpy as np
import pandas as pd

UDF_REGISTRY = {}

TableT = TypeVar("TableT", "Table", np.ndarray, pd.DataFrame)
TensorT = TypeVar("TensorT", "Tensor", np.ndarray)
LoopbackTableT = TypeVar("LoopbackTableT", "LoopbackTable", np.ndarray, pd.DataFrame)
LiteralParameterT = TypeVar("LiteralParameterT", "LiteralParameter", Any)
ScalarT = TypeVar("ScalarT", "Scalar", Any)


def udf(func):
    global UDF_REGISTRY
    module_name = func.__module__.split('.')[-1]
    func_name = func.__name__
    qualname = module_name + '.' + func_name
    UDF_REGISTRY[qualname] = func
    return func

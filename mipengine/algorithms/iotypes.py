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
    UDF_REGISTRY[func.__name__] = func
    return func

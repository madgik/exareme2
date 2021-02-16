from functools import singledispatch

import numpy as np

from worker.udfgen import LiteralParameter
from worker.udfgen import Tensor


@singledispatch
def zeros(shape):
    raise NotImplementedError


@zeros.register
def _(shape: tuple):
    return np.zeros(shape)


@zeros.register
def _(shape: LiteralParameter):
    return Tensor(dtype=float, shape=shape.value)


@singledispatch
def diag(vec):
    raise NotImplementedError


@diag.register
def _(vec: np.ndarray):
    return np.diag(vec)


@diag.register
def _(vec: Tensor):
    assert len(vec.shape) == 1
    return Tensor(dtype=vec.dtype, shape=(vec.shape[0], vec.shape[0]))

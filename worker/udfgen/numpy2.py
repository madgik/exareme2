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

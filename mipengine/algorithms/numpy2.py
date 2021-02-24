from functools import singledispatch

import numpy as np

from mipengine.algorithms.udfgen.udfparams import LiteralParameter
from mipengine.algorithms.udfgen.udfparams import Table
from mipengine.algorithms.udfgen.udfparams import Tensor


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
    assert vec.ncols == 1
    return Tensor(dtype=vec.dtype, shape=(vec.shape[0], vec.shape[0]))


@diag.register
def _(vec: Table):
    assert vec.ncols == 1
    return Table(dtype=vec.dtype, shape=(vec.shape[0], vec.shape[0]))


@singledispatch
def inv(mat):
    raise NotImplemented


@inv.register
def _(mat: np.ndarray):
    return np.linalg.inv(mat)


@inv.register
def _(mat: Table):
    return mat

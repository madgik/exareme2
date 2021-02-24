from copy import copy
from functools import singledispatch

import numpy as np
from scipy import special

from mipengine.algorithms.udfgen.udfparams import Table
from mipengine.algorithms.udfgen.udfparams import Tensor


@singledispatch
def expit(x):
    raise NotImplementedError


@expit.register
def _(x: np.ndarray):
    return special.expit(x)


@expit.register
def _(x: Tensor):
    return copy(x)


@expit.register
def _(x: Table):
    return copy(x)


@singledispatch
def xlogy(x, y):
    raise NotImplementedError


@xlogy.register
def _(x: np.ndarray, y):
    return special.xlogy(x, y)


@xlogy.register
def _(x: Tensor, y):
    return copy(x)


@xlogy.register
def _(x: Table, y):
    return copy(x)

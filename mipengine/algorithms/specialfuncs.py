from functools import singledispatch
from copy import copy

from scipy import special
import numpy as np

from worker.udfgen.udfparams import Tensor


@singledispatch
def expit(x):
    raise NotImplementedError


@expit.register
def _(x: np.ndarray):
    return special.expit(x)


@expit.register
def _(x: Tensor):
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

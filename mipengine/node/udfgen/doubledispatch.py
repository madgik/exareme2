from functools import singledispatch, wraps
from numbers import Number

import numpy as np

from mipengine.algorithms.arraybundle import ArrayBundle, compute_out_type


def doubledispatch(func):

    @wraps(func)
    @singledispatch
    def wrapper(arg1, arg2, *args, **kwargs):
        raise NotImplementedError


@singledispatch
def func(x, y):
    raise NotImplementedError('first')


@func.register(int)
def _(x, y):
    @singledispatch
    def second(y):
        raise NotImplementedError('second')
    @second.register(int)
    def _(y):
        print('int int')
    @second.register(str)
    def _(y):
        print('int str')

    return second(y)

@func.register(str)
def _(x, y):
    @singledispatch
    def second(y):
        raise NotImplementedError
    @second.register(int)
    def _(y):
        print('str int')
    @second.register(str)
    def _(y):
        print('str str')

    return second(y)


@singledispatch
def _ufunc_binary(in1, in2, ufunc, method, **kwargs):
    raise NotImplementedError


@_ufunc_binary.register(ArrayBundle)
def _(arrb, in2, ufunc, method, **kwargs):
    out_type = compute_out_type(ufunc, arrb, in2)
    @singledispatch
    def second(in2):
        raise NotImplementedError
    @second.register(Number)
    def _(num):
        out = np.empty(arrb.shape, dtype=out_type)
        for i, column in enumerate(arrb._itercolumns()):
            getattr(ufunc, method)(num, column, out=out[:, i], **kwargs)
        return out
    @second.register(np.ndarray)
    def _(arr):
        pass  # TODO broadcasting
    @second.register(ArrayBundle)
    def _(arrb):
        pass  # TODO broadcasting


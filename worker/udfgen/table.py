from numbers import Number

import numpy as np

from ufunctypes import type_conversion_table

__all__ = ["Table"]


class Table(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, dtype, shape):
        self.dtype = dtype
        self.shape = shape

    def __repr__(self):
        clsname = type(self).__name__
        return f"{clsname}(dtype={self.dtype.__name__}, shape={self.shape})"

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if not all(isinstance(inpt, (type(self), Number)) for inpt in inputs):
            raise TypeError("Can only apply ufunc between MockTable and Number")
        if ufunc.__name__ == "matmul":
            if inputs[0].shape[1] != inputs[1].shape[0]:
                raise ValueError("Matrix dimensions missmatch")
            newshape = inputs[0].shape[0], inputs[1].shape[1]
            intypes = tuple([_typeof(inpt) for inpt in inputs])
            newtype = type_conversion_table[ufunc.__name__][intypes]
            return Table(dtype=newtype, shape=newshape)
        else:
            if ufunc.nin == 1:
                shape_a = inputs[0].shape
                newshape = shape_a
            elif ufunc.nin == 2:
                if isinstance(inputs[0], Number):
                    inputs = (np.array(inputs[0]), inputs[1])
                if isinstance(inputs[1], Number):
                    inputs = (inputs[0], np.array(inputs[1]))
                shape_a = inputs[0].shape
                shape_b = inputs[1].shape
                newshape = _broadcast_shapes(shape_a, shape_b)
            else:
                raise ValueError("ufuncs do not accept more than 2 operands")
            intypes = tuple([_typeof(inpt) for inpt in inputs])
            newtype = type_conversion_table[ufunc.__name__][intypes]
            return Table(dtype=newtype, shape=newshape)

    def __getitem__(self, key):
        mock = np.broadcast_to(np.array(0), self.shape)
        newshape = mock[key].shape
        if newshape == ():
            newshape = (1,)
        return Table(dtype=self.dtype, shape=newshape)

    @property
    def transpose(self):
        return Table(dtype=self.dtype, shape=(self.shape[1], self.shape[0]))

    T = transpose


def _broadcast_shapes(*shapes):
    """Copied from https://stackoverflow.com/a/54860994/10132636"""
    ml = max(shapes, key=len)
    out = list(ml)
    for s in shapes:
        if s is ml:
            continue
        for i, x in enumerate(s, -len(s)):
            if x != 1 and x != out[i]:
                if out[i] != 1:
                    msg = f"Can't broadcast, dimensions missmatch: {x}, {out[i]}"
                    raise ValueError(msg)
                out[i] = x
    return (*out,)


def _typeof(obj):
    try:
        return obj.dtype.type
    except AttributeError:
        return obj.dtype


# -------------------------------------------------------- #
# Examples                                                 #
# -------------------------------------------------------- #
t = Table(dtype=int, shape=(100, 10))
x = t[0:3]
y = t[1, 2] + t[2, 4]

from numbers import Number

import numpy as np

from ufunctypes import type_conversion_table

__all__ = ["Table"]


class NRows:
    def __repr__(self):
        return "nrows"


NROWS = NRows()


class Table(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, tp, ncolumns=None, shape=None):
        self.type = tp
        if ncolumns and not shape:
            self.shape = (NROWS, ncolumns)
        elif not ncolumns and shape:
            self.shape = shape
        else:
            ValueError(f"Can't instantiate MockTable with {ncolumns} and {shape}")

    def __repr__(self):
        clsname = type(self).__name__
        return f"{clsname}(tp={self.type.__name__}, shape={self.shape})"

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if not all(isinstance(inpt, (type(self), Number)) for inpt in inputs):
            raise TypeError("Can only apply ufunc between MockTable and Number")
        if ufunc.__name__ == "matmul":
            if inputs[0].shape[1] != inputs[1].shape[0]:
                raise ValueError("Matrix dimensions missmatch")
            newshape = inputs[0].shape[0], inputs[1].shape[1]
            intypes = tuple([_typeof(inpt) for inpt in inputs])
            newtype = type_conversion_table[ufunc.__name__][intypes]
            return Table(tp=newtype, shape=newshape)
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
            return Table(tp=newtype, shape=newshape)

    @property
    def transpose(self):
        return Table(tp=self.type, shape=(self.shape[1], self.shape[0]))

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
    if hasattr(obj, "dtype"):
        return obj.dtype.type
    elif hasattr(obj, "type"):
        return obj.type

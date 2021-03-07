from __future__ import annotations
from numbers import Number

import numpy as np

from mipengine.node.udfgen.ufunctypes import get_ufunc_type_conversions

TYPE_CONVERSIONS = get_ufunc_type_conversions()

SQLTYPES = {
    int: "BIGINT",
    float: "DOUBLE",
    str: "TEXT",
    np.int32: "INT",
    np.int64: "BIGINT",
    np.float32: "FLOAT",
    np.float64: "DOUBLE",
}


class DatalessArray(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, dtype, shape):
        self.dtype = dtype
        self.shape = shape

    @property
    def ncols(self):
        if len(self.shape) == 1:
            return 1
        else:
            return self.shape[1]

    def __repr__(self):
        clsname = type(self).__name__
        return f"{clsname}(dtype={self.dtype.__name__}, shape={self.shape})"

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == "__call__":
            inputs = [
                inpt.value if isinstance(inpt, LiteralParameter) else inpt
                for inpt in inputs
            ]
            if not all(
                isinstance(inpt, (DatalessArray, Number, np.ndarray)) for inpt in inputs
            ):
                raise TypeError(
                    "Can only apply ufunc between Table and Number and arrays"
                )
            if ufunc.__name__ == "matmul":
                if type(inputs[0]) == np.ndarray:
                    inputs[0] = DatalessArray(float, inputs[0].shape)
                if type(inputs[1]) == np.ndarray:
                    inputs[1] = DatalessArray(float, inputs[1].shape)
                if inputs[0].shape[-1] != inputs[1].shape[0]:
                    raise ValueError("Matrix dimensions missmatch")
                newshape = inputs[0].shape[:1] + inputs[1].shape[1:]
                intypes = tuple([_typeof(inpt) for inpt in inputs])
                newtype = TYPE_CONVERSIONS[ufunc.__name__][intypes]
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
                newtype = TYPE_CONVERSIONS[ufunc.__name__][intypes]

            out = DatalessArray(dtype=newtype, shape=newshape)
            # if _is_scalar(out):
            #     return Scalar(dtype=newtype)
            return out
        elif method == "reduce":
            # TODO implement
            tab = inputs[0]
            return DatalessArray(dtype=tab.dtype, shape=(1,))

    def __getitem__(self, key):
        mock = np.broadcast_to(np.array(0), self.shape)
        newshape = mock[key].shape
        if newshape == ():
            Scalar(dtype=self.dtype)
        return DatalessArray(dtype=self.dtype, shape=newshape)

    def __len__(self):
        if len(self.shape) == 1:
            return self.shape[0]
        return self.shape[1]

    @property
    def transpose(self):
        if len(self.shape) == 1:
            return self
        return DatalessArray(dtype=self.dtype, shape=(self.shape[1], self.shape[0]))

    T = transpose

    def as_sql_parameters(self, name):
        return ", ".join(
            [f"{name}{_} {SQLTYPES[self.dtype]}" for _ in range(self.ncols)]
        )

    def as_sql_return_type(self, name):
        if _is_scalar(self):
            return SQLTYPES[self.dtype]
        else:
            return f"Table({self.as_sql_parameters(name)})"


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


def _is_scalar(table):
    return all(dim == 1 for dim in table.shape)


def _typeof(obj):
    try:
        return obj.dtype.type
    except AttributeError:
        return obj.dtype


# class Tensor(DatalessArray):
#     pass


class LiteralParameter:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        clsname = type(self).__name__
        return f"{clsname}({self.value})"


class LoopbackTable(DatalessArray):
    def __init__(self, name, dtype, shape):
        self.name = name
        super().__init__(dtype, shape)

    def __repr__(self):
        clsname = type(self).__name__
        dtypename = self.dtype.__name__
        return f'{clsname}(name="{self.name}", dtype={dtypename}, shape={self.shape})'


class Scalar:
    def __init__(self, dtype):
        self.dtype = dtype

    def as_sql_return_type(self):
        return SQLTYPES[self.dtype]

from __future__ import annotations

from numbers import Number

import numpy as np

from worker.udfgen.ufunctypes import type_conversion_table

SQLTYPES = {
    int: "BIGINT",
    float: "DOUBLE",
    str: "TEXT",
    np.int32: "INT",
    np.int64: "BIGINT",
    np.float32: "FLOAT",
    np.float64: "DOUBLE",
}


class Table(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, dtype, shape):
        self.dtype = dtype
        self.shape = shape

    def __repr__(self):
        clsname = type(self).__name__
        return f"{clsname}(dtype={self.dtype.__name__}, shape={self.shape})"

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        inputs = [
            inpt.value if isinstance(inpt, LiteralParameter) else inpt
            for inpt in inputs
        ]
        if not all(isinstance(inpt, (Table, Number)) for inpt in inputs):
            raise TypeError("Can only apply ufunc between Table and Number")
        if ufunc.__name__ == "matmul":
            if inputs[0].shape[-1] != inputs[1].shape[0]:
                raise ValueError("Matrix dimensions missmatch")
            newshape = inputs[0].shape[:1] + inputs[1].shape[1:]
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

    def __len__(self):
        if len(self.shape) == 1:
            return self.shape[0]
        return self.shape[1]

    @property
    def transpose(self):
        if len(self.shape) == 1:
            return self
        return Table(dtype=self.dtype, shape=(self.shape[1], self.shape[0]))

    T = transpose

    def as_sql_parameters(self, name):
        return ", ".join(
            [f"{name}{_} {SQLTYPES[self.dtype]}" for _ in range(self.shape[1])]
        )

    def as_sql_return_declaration(self, name):
        if self.shape == (1,):
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


def _typeof(obj):
    try:
        return obj.dtype.type
    except AttributeError:
        return obj.dtype


class Tensor(Table):
    pass


class LiteralParameter:
    def __init__(self, value):
        self.value = value


class LoopbackTable(Table):
    def __init__(self, name, dtype, shape):
        self.name = name
        super().__init__(dtype, shape)

    def __repr__(self):
        clsname = type(self).__name__
        dtypename = self.dtype.__name__
        return f'{clsname}(name="{self.name}", dtype={dtypename}, shape={self.shape})'

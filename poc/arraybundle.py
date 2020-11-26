from numbers import Number

import numpy as np

from dispatcher import dispatch

# TODO make ArrayBundle.transpose lazy and take advantage of lazyness in X.T @ X

BOOLEAN_UFUNCS = (
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "not_equal",
    "equal",
    "logical_or",
    "logical_and",
    "logical_xor",
    "logical_not",
)


class ArrayBundle(np.lib.mixins.NDArrayOperatorsMixin):
    """Container class for a list of 1-dimensional numpy arrays. Objects of
    this class behave as if the 1-dimensional arrays were columns in a
    2-dimensional numpy array. All ufuncs implementations sacrifice performance
    to save memory, as we keep the ArrayBundle structure (python list of numpy
    arrays) durring the computation instead of copying everything to a new
    array and then do the computation.

    Examples:
    ---------
        >>> arrays = [numpy.array([1., 2., 3.]), numpy.array([10., 20., 30.])]
        >>> X = ArrayBundle(arrays)
        >>> X + 1
        array([[ 2.,  3.,  4.],
               [11., 21., 31.]])
        >>> X.T @ X
        array([[  14.,  140.],
               [ 140., 1400.]])
    """

    _HANDLED_TYPES = (np.ndarray, Number)

    def __init__(self, arrays):
        shape = arrays[0].shape
        assert len(shape) == 1, "arrays must be one-dimensional"
        assert all(
            array.shape == shape for array in arrays
        ), "arrays must all have the same size"
        self.shape = shape[0], len(arrays)  # nrows, ncols
        self._arrays = arrays
        self._HANDLED_TYPES += (self.__class__,)

    @property
    def transpose(self):  # TODO make lazy
        self_as_array = np.array(self)
        return self_as_array.T

    T = transpose

    def copy(self):
        return np.array(self)

    def _itercolumns(self):
        return iter(self._arrays)

    def __iter__(self):
        return (
            np.array([array[col] for col in range(self.shape[1])])
            for array in self._itercolumns()
        )

    def __len__(self):
        return self.shape[0]

    def __str__(self):
        """
        >>> print(ArrayBundle([np.array([1, 2, 3]), np.array([10, 20, 30])]))
        [[ 1 10]
         [ 2 20]
         [ 3 30]]

        This is counterintuitive but, contrary to numpy arrays, here each
        subarray in the constructor corresponds to one column, not one row.
        """
        return str(np.array(self))

    def __repr__(self):
        class_name = type(self).__name__
        indent = len(class_name) + 2
        lines = [indent * " " + repr(array) + "," for array in self._itercolumns()]
        lines[0] = f"{class_name}([" + lines[0][indent:]
        lines[-1] = lines[-1][:-1] + "])"
        return "\n".join(lines)

    def __array__(self):
        return np.array([array for array in self._itercolumns()]).T

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Most important method of the class. Here we implement the logic
        behind ufunc calls to ArrayBundle objects. All ufunc methods are
        implemented except inplace ones. In all cases, memory load is minimized
        by only allocating a new numpy array for the result.

        For all binary ufuncs, all operand type combinations are handled by a
        multiple dispatch mechanism.
        """
        out = kwargs.get("out", ())
        if out:
            raise ValueError(f"`out` cannot be used with {type(self).__name__}")
        for x in inputs + out:
            if not isinstance(x, self._HANDLED_TYPES):
                return NotImplemented
        if method in ("__call__", "reduce", "accumulate", "outer"):
            if len(inputs) == 1:
                operand = inputs[0]
                out = np.empty(operand.shape)
                for i, column in enumerate(self._itercolumns()):
                    getattr(ufunc, method)(column, out=out[:, i], **kwargs)
                return out
            elif len(inputs) == 2 and ufunc.__name__ != "matmul":
                return _ufunc_binary(inputs, ufunc, method, **kwargs)
            elif len(inputs) == 2 and ufunc.__name__ == "matmul":
                return _ufunc_matmul(inputs, ufunc, method, **kwargs)
            else:
                raise ValueError("проблема!")
        elif method in ("at", "reduceat"):
            raise NotImplementedError

    def __getitem__(self, index):
        """Implements the usual indexing of numpy arrays (partly) where index
        can be an integer, a slice, a tuple of ints and/or slices or an int or
        bool array.
        """
        if type(index) == int:
            if not (0 <= index < self.shape[0]):
                raise IndexError("index out of bounds")
            return np.array([array[index] for array in self._itercolumns()])
        elif type(index) == tuple:
            if len(index) > len(self.shape):
                raise IndexError("too many indices for array")
            rowidx, colidx = index
            return self._arrays[colidx][rowidx]
        elif type(index) == np.ndarray and index.dtype in (np.int, np.bool):
            return np.array(self)[index]
        else:
            raise IndexError(
                "only integers, slices and int or bool arrays are valid indices"
            )

    def __setitem__(self, key, value):
        class_name = type(self).__name__
        raise TypeError(
            f"{class_name} objects don't support assignment. "
            "Use ufunc with `at` method for inplace processing "
            "or use `copy` method instead."
        )

    def __delitem__(self, key):
        class_name = type(self).__name__
        raise TypeError(f"{class_name} objects don't support deletion")


# -------------------------------------------------------------------------------- #
# Dispatch methods                                                                 #
# -------------------------------------------------------------------------------- #
@dispatch
def _ufunc_binary(inputs, ufunc, method, **kwargs):
    raise NotImplementedError


@_ufunc_binary.register(Number, ArrayBundle)
def _(inputs, ufunc, method, **kwargs):
    num, arrb = inputs
    out_type = comput_out_type(ufunc, *inputs)
    out = np.empty(arrb.shape, dtype=out_type)
    for i, column in enumerate(arrb._itercolumns()):
        getattr(ufunc, method)(num, column, out=out[:, i], **kwargs)
    return out


@_ufunc_binary.register(ArrayBundle, Number)
def _(inputs, ufunc, method, **kwargs):
    arrb, num = inputs
    out_type = comput_out_type(ufunc, *inputs)
    out = np.empty(arrb.shape, dtype=out_type)
    for i, column in enumerate(arrb._itercolumns()):
        getattr(ufunc, method)(column, num, out=out[:, i], **kwargs)
    return out


@_ufunc_binary.register(np.ndarray, ArrayBundle)
def _(inputs, ufunc, method, **kwargs):
    arr, arrb = inputs
    assert arr.shape == arrb.shape, "For now no broadcasting"
    out_type = comput_out_type(ufunc, *inputs)
    out = np.empty(arr.shape, dtype=out_type)
    for i, column in enumerate(arrb._itercolumns()):
        getattr(ufunc, method)(arr[:, i], column, out=out[:, i], **kwargs)
    return out


@_ufunc_binary.register(ArrayBundle, np.ndarray)
def _(inputs, ufunc, method, **kwargs):
    arrb, arr = inputs
    assert arr.shape == arrb.shape, "For now no broadcasting"
    out_type = comput_out_type(ufunc, *inputs)
    out = np.empty(arr.shape, dtype=out_type)
    for i, column in enumerate(arrb._itercolumns()):
        getattr(ufunc, method)(column, arr[:, i], out=out[:, i], **kwargs)
    return out


@_ufunc_binary.register(ArrayBundle, ArrayBundle)
def _(inputs, ufunc, method, **kwargs):
    arrb_1, arrb_2 = inputs
    assert arrb_1.shape == arrb_2.shape, "For now no broadcasting"
    out_type = comput_out_type(ufunc, *inputs)
    out = np.empty(arrb_1.shape, dtype=out_type)
    columns_1, columns_2 = arrb_1._itercolumns(), arrb_2._itercolumns()
    for i, (col_1, col_2) in enumerate(zip(columns_1, columns_2)):
        getattr(ufunc, method)(col_1, col_2, out=out[:, i], **kwargs)
    return out


def comput_out_type(ufunc, op_1, op_2):
    if ufunc.__name__ in BOOLEAN_UFUNCS:
        return bool
    if type(op_1) == int and type(op_2) == int:
        return int
    return float


@dispatch
def _ufunc_matmul(inputs, ufunc, method, **kwargs):
    raise NotImplementedError


@_ufunc_matmul.register(ArrayBundle, np.ndarray)
def _(inputs, ufunc, method, **kwargs):
    arrb_l, arr_r = inputs
    if arrb_l.shape[1] != arr_r.shape[0]:
        raise ValueError("matmul: operand dimension mismatch")
    out_type = comput_out_type(ufunc, *inputs)
    out = np.empty((arrb_l.shape[0], arr_r.shape[1]), dtype=out_type)
    dimleft = arrb_l.shape[0]
    dimmid = arrb_l.shape[1]
    dimright = arr_r.shape[1]
    for i in range(dimleft):
        for j in range(dimright):
            out[i, j] = sum(arrb_l[i, k] * arr_r[k, j] for k in range(dimmid))
    return out


@_ufunc_matmul.register(np.ndarray, ArrayBundle)
def _(inputs, ufunc, method, **kwargs):
    arr_l, arr_bund_r = inputs
    if arr_l.shape[1] != arr_bund_r.shape[0]:
        raise ValueError("matmul: operand dimension mismatch")
    out_type = comput_out_type(ufunc, *inputs)
    out = np.empty((arr_l.shape[0], arr_bund_r.shape[1]), dtype=out_type)
    dimleft = arr_l.shape[0]
    dimright = arr_bund_r.shape[1]
    for i in range(dimleft):
        for j in range(dimright):
            out[i, j] = arr_l[i] @ arr_bund_r[:, j]
    return out


@_ufunc_matmul.register(ArrayBundle, ArrayBundle)
def _(inputs, ufunc, method, **kwargs):
    arrb_l, arrb_r = inputs
    if arrb_l.shape[1] != arrb_r.shape[0]:
        raise ValueError("matmul: operand dimension mismatch")
    out_type = comput_out_type(ufunc, *inputs)
    out = np.empty((arrb_l.shape[0], arrb_r.shape[1]), dtype=out_type)
    dimleft = arrb_l.shape[0]
    dimmid = arrb_l.shape[1]
    dimright = arrb_r.shape[1]
    for i in range(dimleft):
        for j in range(dimright):
            out[i, j] = sum(arrb_l[i, k] * arrb_r[k, j] for k in range(dimmid))
    return out


if __name__ == "__main__":
    xb = ArrayBundle([np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0])])
    print("repr:")
    print(repr(xb), end="\n\n")

    print("str:")
    print(xb, end="\n\n")

    print("unary ufunc")
    print(np.sin(xb), end="\n\n")

    print("op with ndarray:")
    x = np.array([[1, 2], [10, 20], [100, 200]])
    print(xb * x)
    print(x * xb, end="\n\n")

    print("fancy indexing:")
    print(xb[xb > 3], end="\n\n")

    print("op with other ArrayBundle:")
    yb = ArrayBundle([np.array([1, 2, 3]), np.array([10, 20, 30])])
    print(xb * yb, end="\n\n")
    print(yb * xb, end="\n\n")

    print("matmul with ndarray:")
    print(xb.T @ x, end="\n\n")
    print(x.T @ xb, end="\n\n")

    print("matmul with ArrayBundle:")
    print(xb.T @ yb, end="\n\n")
    print(yb.T @ xb, end="\n\n")

import numbers

import numpy as np


class ArrayBundle(np.lib.mixins.NDArrayOperatorsMixin):
    """Container class for a list of 1-dimensional numpy arrays. Objects of
    this class behave as if the 1-dimensional arrays were columns in a
    2-dimensional numpy array. To reduce memory load all in-place operations
    keep the 1-dimensional arrays in their original locations.  However, for
    all operations that would return a copy anyway, the object is cast into a
    2-dimensional numpy array to simplify things.

    The example below shows how __add__ returns a new numpy array whereas
    np.log.at takes the log elementwise and in-place.

    Examples:
    ---------
        >>> arrays = [numpy.array([1., 2., 3.]), numpy.array([10., 20., 30.])]
        >>> X = ArrayBundle(arrays)
        >>> X + 1
        array([[ 2.,  3.,  4.],
               [11., 21., 31.]])
        >>> np.log.at(X, True)
        >>> X
        ArrayBundle(array([0.        , 0.69314718, 1.09861229])
                    array([2.30258509, 2.99573227, 3.40119738]))

    """

    _HANDLED_TYPES = (np.ndarray, numbers.Number)

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
    def transpose(self):
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
        behind ufunc calls to ArrayBundle objects. All ufunc method types
        except `at` have a simple implementation where the current object is
        cast into a numpy ndarray and the ufunc is called afrerwards.  `at`
        methods are more involved since they have to apply the ufunc in-place
        to avoid memory load. Currently only unary `at` methods are supported.
        """
        out = kwargs.get("out", ())
        if out:
            raise ValueError(f'`out` cannot be used with {type(self).__name__}')
        for x in inputs + out:
            if not isinstance(x, self._HANDLED_TYPES):
                return NotImplemented
        if method in ("__call__", "reduce", "accumulate", "outer"):
            # all these methods return a copy anyway so we cast all
            # ArrayBundles as numpy ndarrays and then call the ufunc
            inputs = tuple(
                np.array(x) if isinstance(x, self.__class__) else x for x in inputs
            )
            return getattr(ufunc, method)(*inputs, **kwargs)
        elif method == "at":
            # `at` methods perform in place processing so are treated
            # differently
            if len(inputs) == 2:
                _, indices = inputs
                for array in self._itercolumns():
                    getattr(ufunc, method)(array, indices)
                return None
            elif len(inputs) == 3:
                raise NotImplementedError
        else:
            # `reduceat` not implemented
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


if __name__ == "__main__":
    xb = ArrayBundle([np.array([1., 2., 3.]), np.array([10., 20., 30.])])
    print('repr:')
    print(repr(xb), end='\n\n')

    print('str:')
    print(xb, end='\n\n')

    print('op with ndarray:')
    x = np.array([[1, 2], [10, 20], [100, 200]])
    print(xb * x, end='\n\n')

    print('fancy indexing:')
    print(xb[xb > 0], end='\n\n')

    print('op with other ArrayBundle:')
    yb = ArrayBundle([np.array([1, 2, 3]), np.array([10, 20, 30])])
    print(xb.T @ yb, end='\n\n')

    print('inplace op:')
    np.sin.at(xb, True)
    print(repr(xb), end='\n\n')

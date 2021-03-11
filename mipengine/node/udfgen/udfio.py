import re

import numpy as np


def as_tensor_table(array: np.ndarray):
    size = array.size
    shape = array.shape
    indices = np.unravel_index(range(size), shape)
    out = {f"dim{i}": idx for i, idx in enumerate(indices)}
    out["val"] = array.ravel()
    return out


def from_tensor_table(table: dict):
    # XXX Hack, find better way
    table = {
        re.sub(r"\w+_(dim\d+)", r"\1", key)
        if re.match(r"\w+_(dim\d+)", key)
        else re.sub(r"\w+_(val)", r"\1", key): val
        for key, val in table.items()
    }
    ndims = len(table) - 1
    multi_index = [table[f"dim{i}"] for i in range(ndims)]
    shape = [max(idx) + 1 for idx in multi_index]
    lin_index = np.ravel_multi_index(multi_index, shape)
    if all(li == i for i, li in enumerate(lin_index)):
        array = table["val"].reshape(shape)
    else:
        array = table["val"][lin_index].reshape(shape)
    return np.array(array)


def as_relational_table(array: np.ndarray, name: str):
    assert len(array.shape) in (1, 2)
    names = (f"{name}_{i}" for i in range(array.shape[1]))
    out = {n: c for n, c in zip(names, array)}
    return out

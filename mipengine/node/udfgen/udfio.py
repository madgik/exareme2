from functools import partial, reduce
import re

import numpy as np
import pandas as pd


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


def reduce_tensor_pair(op, a: pd.DataFrame, b: pd.DataFrame):
    ndims = len(a.columns) - 1
    dimensions = [f"dim{_}" for _ in range(ndims)]
    merged = a.merge(b, left_on=dimensions, right_on=dimensions)
    merged["val"] = merged.apply(lambda df: op(df.val_x, df.val_y), axis=1)
    return merged[dimensions + ["val"]]


def reduce_tensor_merge_table(op, merge_table):
    groups = [group for _, group in merge_table.groupby("node_id")]
    groups = [group.drop("node_id", 1) for group in groups]
    result = reduce(partial(reduce_tensor_pair, op), groups)
    return result


def make_tensor_merge_table(columns):
    colnames = columns.keys()
    expected = {"node_id", "val"}
    if expected - set(colnames):
        raise ValueError("No node_id or val in columns.")
    columns = {n: v for n, v in columns.items() if n in expected or n.startswith("dim")}
    if len(columns) <= 2:
        raise ValueError(f"Columns have wrong format {columns}.")
    return pd.DataFrame(columns)

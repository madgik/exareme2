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


def merge_tensor_to_list(columns):
    colnames = list(columns.keys())
    try:
        node_id_column_idx = next(
            i for i, colname in enumerate(colnames) if re.match(r".*node_id", colname)
        )
    except StopIteration:
        raise ValueError("No column is named .*node_id")
    node_id_name = colnames[node_id_column_idx]
    merge_df = pd.DataFrame(columns)
    groups = [group for _, group in merge_df.groupby(node_id_name)]
    groups = [group.drop(node_id_name, 1) for group in groups]
    all_cols = [
        {colname: np.array(x) for colname, x in df.to_dict(orient="list").items()}
        for df in groups
    ]
    xs = [from_tensor_table(cols) for cols in all_cols]
    return xs


def secure_transfers_to_merged_dict(transfers):
    """
    Converts a list of secure transfer dictionaries to one dictionary that
    contains the aggregation of all the initial values.
    """
    # TODO Should also work for "decimals", "min", "max" and "union"

    result = {}

    # Get all keys from a list of dicts
    all_keys = set().union(*(d.keys() for d in transfers))
    for key in all_keys:
        operation = transfers[0][key]["operation"]
        op_type = transfers[0][key]["type"]
        if operation == "addition":
            if op_type == "int":
                result[key] = _add_secure_transfer_key_integer_data(key, transfers)
            else:
                raise NotImplementedError(
                    f"Secure transfer type: {type} not supported for operation: {operation}"
                )
        else:
            raise NotImplementedError(
                f"Secure transfer operation not supported: {operation}"
            )
    return result


def _add_secure_transfer_key_integer_data(key, transfers):
    """
    Given a list of secure_transfer dicts, it sums the data of the key provided.
    The values should be integers.
    """
    result = transfers[0][key]["data"]
    for transfer in transfers[1:]:
        if transfer[key]["operation"] != "addition":
            raise ValueError(
                f"All secure transfer keys should have the same 'operation' value. 'addition' != {transfer[key]['operation']}"
            )
        if transfer[key]["type"] != "int":
            raise ValueError(
                f"All secure transfer keys should have the same 'type' value. 'int' != {transfer[key]['type']}"
            )
        result = _add_integer_type_values(result, transfer[key]["data"])
    return result


def _add_integer_type_values(value1, value2):
    """
    The values could be either integers or lists that contain other lists or integers.
    The type of the values should not change, only the value.
    """
    if type(value1) != type(value2):
        raise TypeError(
            f"Secure transfer data have different types: {type(value1)} != {type(value2)}"
        )

    if isinstance(value1, list):
        result = []
        for e1, e2 in zip(value1, value2):
            result.append(_add_integer_type_values(e1, e2))
        return result
    else:
        return value1 + value2

import logging
import os
import re
from functools import partial
from functools import reduce
from typing import Any
from typing import List
from typing import Tuple
from typing import Type

import numpy as np
import pandas as pd

LOG_LEVEL_ENV_VARIABLE = "LOG_LEVEL"
LOG_LEVEL_DEFAULT_VALUE = "INFO"


def get_logger(udf_name: str, request_id: str):
    logger = logging.getLogger("monetdb_udf")
    for handler in logger.handlers:
        logger.removeHandler(handler)

    log_level = os.getenv(LOG_LEVEL_ENV_VARIABLE, LOG_LEVEL_DEFAULT_VALUE)
    formatter = logging.Formatter(
        f"%(asctime)s - %(levelname)s - MONETDB - PYTHONUDF - {udf_name}(%(lineno)d) - {request_id} - %(message)s"
    )
    # StreamHandler
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.setLevel(log_level)
    return logger


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


def as_relational_table(array: np.ndarray):
    """
    TODO Output of relational tables needs to be refactored
    What objects can we return? Do we need a name parameter?
    https://team-1617704806227.atlassian.net/browse/MIP-536

    """
    # assert len(array.shape) in (1, 2)
    # names = (f"{name}_{i}" for i in range(array.shape[1]))
    # out = {n: c for n, c in zip(names, array)}
    return array


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


# ~~~~~~~~~~~~~~~~~~~~~~~~ Secure Transfer methods ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


numeric_operations = ["sum", "min", "max"]


def secure_transfers_to_merged_dict(transfers: List[dict]):
    """
    Converts a list of secure transfer dictionaries to one dictionary that
    contains the aggregation of all the initial values.
    This is used for secure transfer objects when SMPC is disabled.
    """
    result = {}

    # Get all keys from a list of dicts
    all_keys = set().union(*(d.keys() for d in transfers))
    for key in all_keys:
        operation = transfers[0][key]["operation"]
        if operation in numeric_operations:
            result[key] = _operation_on_secure_transfer_key_data(
                key, transfers, operation
            )
        else:
            raise NotImplementedError(
                f"Secure transfer operation not supported: {operation}"
            )
    return result


def _operation_on_secure_transfer_key_data(key, transfers: List[dict], operation: str):
    """
    Given a list of secure_transfer dicts, it makes the appropriate operation on the data of the key provided.
    """
    result = transfers[0][key]["data"]
    for transfer in transfers[1:]:
        if transfer[key]["operation"] not in numeric_operations:
            raise ValueError(
                f"Secure Transfer operation is not supported: {transfer[key]['operation']}"
            )
        if transfer[key]["operation"] != operation:
            raise ValueError(
                f"All secure transfer keys should have the same 'operation' value. "
                f"'{operation}' != {transfer[key]['operation']}"
            )
        result = _calc_values(result, transfer[key]["data"], operation)
    return result


def _calc_values(value1: Any, value2: Any, operation: str):
    """
    The values could be either integers/floats or lists that contain other lists or integers/floats.
    """
    _validate_calc_values(value1, value2)

    if isinstance(value1, list) and isinstance(value2, list):
        result = []
        for e1, e2 in zip(value1, value2):
            result.append(_calc_values(e1, e2, operation))
        return result

    return _calc_numeric_values(value1, value2, operation)


def _validate_calc_values(value1, value2):
    allowed_types = [int, float, list]
    for value in [value1, value2]:
        if type(value) not in allowed_types:
            raise TypeError(
                f"Secure transfer data must have one of the following types: "
                f"{allowed_types}. Type provided: {type(value)}"
            )
    if (isinstance(value1, list) or isinstance(value2, list)) and (
        type(value1) != type(value2)
    ):
        raise ValueError("Secure transfers' data should have the same structure.")


def _calc_numeric_values(value1: Any, value2: Any, operation: str):
    if operation == "sum":
        return value1 + value2
    elif operation == "min":
        return value1 if value1 < value2 else value2
    elif operation == "max":
        return value1 if value1 > value2 else value2
    else:
        raise NotImplementedError


def split_secure_transfer_dict(dict_: dict) -> Tuple[dict, list, list, list, list]:
    """
    When SMPC is used, a secure transfer dict should be split in different parts:
    1) The template of the dict with relative positions instead of values,
    2) flattened lists for each operation, containing the values.
    """
    secure_transfer_template = {}
    op_flat_data = {"sum": [], "min": [], "max": []}
    op_indexes = {"sum": 0, "min": 0, "max": 0}
    for key, data_transfer in dict_.items():
        _validate_secure_transfer_item(key, data_transfer)
        cur_op = data_transfer["operation"]
        try:
            (
                data_transfer_tmpl,
                cur_flat_data,
                op_indexes[cur_op],
            ) = _flatten_data_and_keep_relative_positions(
                op_indexes[cur_op], data_transfer["data"], [list, int, float]
            )
        except TypeError as e:
            raise TypeError(
                f"Secure Transfer key: '{key}', operation: '{cur_op}'. Error: {str(e)}"
            )
        op_flat_data[cur_op].extend(cur_flat_data)

        secure_transfer_template[key] = dict_[key]
        secure_transfer_template[key]["data"] = data_transfer_tmpl

    return (
        secure_transfer_template,
        op_flat_data["sum"],
        op_flat_data["min"],
        op_flat_data["max"],
        [],
    )


def _validate_secure_transfer_item(key: str, data_transfer: dict):
    if "operation" not in data_transfer.keys():
        raise ValueError(
            f"Each Secure Transfer key should contain an operation. Key: {key}"
        )

    if "data" not in data_transfer.keys():
        raise ValueError(f"Each Secure Transfer key should contain data. Key: {key}")

    if data_transfer["operation"] not in numeric_operations:
        raise ValueError(
            f"Secure Transfer operation is not supported: {data_transfer['operation']}"
        )


def construct_secure_transfer_dict(
    template: dict,
    sum_op_values: List[int] = None,
    min_op_values: List[int] = None,
    max_op_values: List[int] = None,
    union_op_values: List[int] = None,
) -> dict:
    """
    When SMPC is used, a secure_transfer dict is broken into template and values.
    In order to be used from a udf it needs to take it's final key - value form.
    """
    final_dict = {}
    for key, data_transfer in template.items():
        if data_transfer["operation"] == "sum":
            unflattened_data = _unflatten_data_using_relative_positions(
                data_transfer["data"], sum_op_values, [int, float]
            )
        elif data_transfer["operation"] == "min":
            unflattened_data = _unflatten_data_using_relative_positions(
                data_transfer["data"], min_op_values, [int, float]
            )
        elif data_transfer["operation"] == "max":
            unflattened_data = _unflatten_data_using_relative_positions(
                data_transfer["data"], max_op_values, [int, float]
            )
        else:
            raise ValueError(f"Operation not supported: {data_transfer['operation']}")

        final_dict[key] = unflattened_data
    return final_dict


def _flatten_data_and_keep_relative_positions(
    index: int,
    data: Any,
    allowed_types: List[Type],
) -> Tuple[Any, List[Any], int]:
    """
    Iterates through a nested list structure and:
    1) keeps the structure of the data with relative positions in the flat array,
    2) flattens the values to a new array and
    3) also returns the final index so it can be used again.

    For example:
    >>> _flatten_data_and_keep_relative_positions(0, [[7,6,7],[8,9,10]])
        Returns:
        [7, 6, 7, 8, 9, 10], [[0, 1, 2], [3, 4, 5]], 6

    """
    if type(data) not in allowed_types:
        raise TypeError(f"Types allowed: {allowed_types}")

    if isinstance(data, list):
        data_pos_template = []
        flat_data = []
        for elem in data:
            (
                data_template,
                cur_flat_data,
                index,
            ) = _flatten_data_and_keep_relative_positions(index, elem, allowed_types)
            data_pos_template.append(data_template)
            flat_data.extend(cur_flat_data)
        return data_pos_template, flat_data, index

    return index, [data], index + 1


def _unflatten_data_using_relative_positions(
    data_tmpl: Any,
    flat_values: List[Any],
    allowed_types: List[Type],
):
    """
    It's doing the exact opposite of _flatten_int_data_and_keep_relative_positions.
    This is used to reconstruct the secure_transfer object after the computations
    from the SMPC.
    """
    if isinstance(data_tmpl, list):
        return [
            _unflatten_data_using_relative_positions(elem, flat_values, allowed_types)
            for elem in data_tmpl
        ]

    if type(data_tmpl) not in allowed_types:
        raise TypeError(f"Types allowed: {allowed_types}")

    return flat_values[data_tmpl]

import logging
import os
import re
from functools import partial
from functools import reduce
from typing import Any
from typing import List
from typing import Set
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
    ndims = len(table) - 1
    multi_index = [table[f"dim{i}"] for i in range(ndims)]
    shape = [max(idx) + 1 for idx in multi_index]
    lin_index = np.ravel_multi_index(multi_index, shape)
    if all(li == i for i, li in enumerate(lin_index)):
        array = table["val"].reshape(shape)
    else:
        array = table["val"][lin_index].reshape(shape)
    return np.array(array)


def from_relational_table(table: dict, row_id: str):
    result = pd.DataFrame(table, copy=False)
    if row_id in result.columns:
        return result.set_index(row_id)
    return result


def as_relational_table(result, row_id):
    if isinstance(result, pd.DataFrame) and row_id == result.index.name:
        return result.reset_index()
    return result


def reduce_tensor_pair(op, a: pd.DataFrame, b: pd.DataFrame):
    ndims = len(a.columns) - 1
    dimensions = [f"dim{_}" for _ in range(ndims)]
    merged = a.merge(b, left_on=dimensions, right_on=dimensions)
    merged["val"] = merged.apply(lambda df: op(df.val_x, df.val_y), axis=1)
    return merged[dimensions + ["val"]]


def reduce_tensor_merge_table(op, merge_table):
    groups = [group for _, group in merge_table.groupby("worker_id")]
    groups = [group.drop("worker_id", 1) for group in groups]
    result = reduce(partial(reduce_tensor_pair, op), groups)
    return result


def make_tensor_merge_table(columns):
    colnames = columns.keys()
    expected = {"worker_id", "val"}
    if expected - set(colnames):
        raise ValueError("No worker_id or val in columns.")
    columns = {n: v for n, v in columns.items() if n in expected or n.startswith("dim")}
    if len(columns) <= 2:
        raise ValueError(f"Columns have wrong format {columns}.")
    return pd.DataFrame(columns)


def merge_tensor_to_list(columns):
    colnames = list(columns.keys())
    try:
        worker_id_column_idx = next(
            i for i, colname in enumerate(colnames) if re.match(r".*worker_id", colname)
        )
    except StopIteration:
        raise ValueError("No column is named .*worker_id")
    worker_id_name = colnames[worker_id_column_idx]
    merge_df = pd.DataFrame(columns)
    groups = [group for _, group in merge_df.groupby(worker_id_name)]
    groups = [group.drop(worker_id_name, axis=1) for group in groups]
    all_cols = [
        {colname: np.array(x) for colname, x in df.to_dict(orient="list").items()}
        for df in groups
    ]
    xs = [from_tensor_table(cols) for cols in all_cols]
    return xs


# ~~~~~~~~~~~~~~~~~~~~~~~~ Secure Transfer methods ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

smpc_sum_op = "sum"
smpc_min_op = "min"
smpc_max_op = "max"
smpc_numeric_operations = [smpc_sum_op, smpc_min_op, smpc_max_op]

smpc_int_type = "int"
smpc_float_type = "float"
smpc_numeric_types = [smpc_int_type, smpc_float_type]
smpc_transfer_op_key = "operation"
smpc_transfer_val_type_key = "type"
smpc_transfer_data_key = "data"


def secure_transfers_to_merged_dict(transfers: List[dict]):
    """
    Converts a list of secure transfer dictionaries to one dictionary that
    contains the aggregation of all the initial values.
    This is used for secure transfer objects when SMPC is disabled.
    """
    result = {}

    # Get all keys from a list of dicts
    all_keys = set().union(*(d.keys() for d in transfers))
    _validate_transfers_have_all_keys(transfers, all_keys)
    for key in all_keys:
        _validate_transfers_operation(transfers, key)
        _validate_transfers_type(transfers, key)
        result[key] = _operation_on_secure_transfer_key_data(
            key,
            transfers,
            transfers[0][key][smpc_transfer_op_key],
        )
    return result


def _validate_transfers_have_all_keys(transfers: List[dict], all_keys: Set[str]):
    for key in all_keys:
        for transfer in transfers:
            if key not in transfer.keys():
                raise ValueError(
                    f"All secure transfer dicts should have the same keys. Transfer: {transfer} doesn't have key: {key}"
                )


def _validate_transfers_operation(transfers: List[dict], key: str):
    """
    Validates that all transfer dicts have proper 'operation' values for the 'key' provided.
    """
    _validate_transfer_key_operation(transfers[0][key])
    first_transfer_operation = transfers[0][key][smpc_transfer_op_key]
    for transfer in transfers[1:]:
        _validate_transfer_key_operation(transfer[key])
        if transfer[key][smpc_transfer_op_key] != first_transfer_operation:
            raise ValueError(
                f"Similar secure transfer keys should have the same operation value. "
                f"'{first_transfer_operation}' != {transfer[key][smpc_transfer_op_key]}"
            )


def _validate_transfer_key_operation(transfer_value: dict):
    try:
        operation = transfer_value[smpc_transfer_op_key]
    except KeyError:
        raise ValueError(
            "Secure Transfer operation is not provided. Expected format: {'a' : {'data': X, 'operation': Y, 'type': Z}}"
        )
    if operation not in smpc_numeric_operations:
        raise ValueError(f"Secure Transfer operation is not supported: '{operation}'.")


def _validate_transfers_type(transfers: List[dict], key: str):
    """
    Validates that all transfer dicts have proper 'type' values for the 'key' provided.
    """
    _validate_transfer_key_type(transfers[0][key])
    first_transfer_value_type = transfers[0][key][smpc_transfer_val_type_key]
    for transfer in transfers[1:]:
        _validate_transfer_key_type(transfer[key])
        if transfer[key][smpc_transfer_val_type_key] != first_transfer_value_type:
            raise ValueError(
                f"Similar secure transfer keys should have the same type value. "
                f"'{first_transfer_value_type}' != {transfer[key][smpc_transfer_val_type_key]}"
            )


def _validate_transfer_key_type(transfer_value: dict):
    try:
        values_type = transfer_value[smpc_transfer_val_type_key]
    except KeyError:
        raise ValueError(
            "Secure Transfer type is not provided. Expected format: {'a' : {'data': X, 'operation': Y, 'type': Z}}"
        )
    if values_type not in smpc_numeric_types:
        raise ValueError(f"Secure Transfer type is not supported: '{values_type}'.")


def _operation_on_secure_transfer_key_data(key, transfers: List[dict], operation: str):
    """
    Given a list of secure_transfer dicts, it makes the appropriate operation on the data of the key provided.
    """
    result = transfers[0][key][smpc_transfer_data_key]
    for transfer in transfers[1:]:
        result = _calc_values(result, transfer[key][smpc_transfer_data_key], operation)
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
    if operation == smpc_sum_op:
        return value1 + value2
    elif operation == smpc_min_op:
        return min(value1, value2)
    elif operation == smpc_max_op:
        return max(value1, value2)
    else:
        raise NotImplementedError


def split_secure_transfer_dict(secure_transfer: dict) -> Tuple[dict, list, list, list]:
    """
    When SMPC is used, a secure transfer dict should be split in different parts:
    1) The template of the dict with relative positions instead of values,
    2) flattened lists for each operation, containing the values.
    """
    secure_transfer_template = {}
    op_flat_data = {smpc_sum_op: [], smpc_min_op: [], smpc_max_op: []}
    op_indexes = {smpc_sum_op: 0, smpc_min_op: 0, smpc_max_op: 0}
    for key, data_transfer in secure_transfer.items():
        _validate_secure_transfer_item(key, data_transfer)
        cur_op = data_transfer[smpc_transfer_op_key]
        try:
            (
                data_transfer_tmpl,
                cur_flat_data,
                op_indexes[cur_op],
            ) = _flatten_data_and_keep_relative_positions(
                op_indexes[cur_op],
                data_transfer[smpc_transfer_data_key],
                [list, int, float],
            )
        except TypeError as e:
            raise TypeError(
                f"Secure Transfer key: '{key}', operation: '{cur_op}'. Error: {str(e)}"
            )
        op_flat_data[cur_op].extend(cur_flat_data)

        secure_transfer_key_template = {
            smpc_transfer_op_key: secure_transfer[key][smpc_transfer_op_key],
            smpc_transfer_val_type_key: secure_transfer[key][
                smpc_transfer_val_type_key
            ],
            smpc_transfer_data_key: data_transfer_tmpl,
        }
        secure_transfer_template[key] = secure_transfer_key_template
    return (
        secure_transfer_template,
        op_flat_data[smpc_sum_op],
        op_flat_data[smpc_min_op],
        op_flat_data[smpc_max_op],
    )


def _validate_secure_transfer_item(key: str, data_transfer: dict):
    if smpc_transfer_op_key not in data_transfer.keys():
        raise ValueError(
            f"Each Secure Transfer key should contain an operation. Key: {key}"
        )

    if smpc_transfer_val_type_key not in data_transfer.keys():
        raise ValueError(f"Each Secure Transfer key should contain a type. Key: {key}")

    if smpc_transfer_data_key not in data_transfer.keys():
        raise ValueError(f"Each Secure Transfer key should contain data. Key: {key}")

    if data_transfer[smpc_transfer_op_key] not in smpc_numeric_operations:
        raise ValueError(
            f"Secure Transfer operation is not supported: {data_transfer[smpc_transfer_op_key]}"
        )

    if data_transfer[smpc_transfer_val_type_key] not in smpc_numeric_types:
        raise ValueError(
            f"Secure Transfer type is not supported: {data_transfer[smpc_transfer_val_type_key]}"
        )


def construct_secure_transfer_dict(
    template: dict,
    sum_op_values: List[int] = None,
    min_op_values: List[int] = None,
    max_op_values: List[int] = None,
) -> dict:
    """
    When SMPC is used, a secure_transfer dict is broken into template and values.
    In order to be used from a udf it needs to take it's final key - value form.
    """
    final_dict = {}
    for key, data_transfer_tmpl in template.items():
        if data_transfer_tmpl[smpc_transfer_op_key] == smpc_sum_op:
            structured_data = _structure_data_using_relative_positions(
                data_transfer_tmpl[smpc_transfer_data_key],
                data_transfer_tmpl[smpc_transfer_val_type_key],
                sum_op_values,
                [int, float],
            )
        elif data_transfer_tmpl[smpc_transfer_op_key] == smpc_min_op:
            structured_data = _structure_data_using_relative_positions(
                data_transfer_tmpl[smpc_transfer_data_key],
                data_transfer_tmpl[smpc_transfer_val_type_key],
                min_op_values,
                [int, float],
            )
        elif data_transfer_tmpl[smpc_transfer_op_key] == smpc_max_op:
            structured_data = _structure_data_using_relative_positions(
                data_transfer_tmpl[smpc_transfer_data_key],
                data_transfer_tmpl[smpc_transfer_val_type_key],
                max_op_values,
                [int, float],
            )
        else:
            raise ValueError(
                f"Operation not supported: {data_transfer_tmpl[smpc_transfer_op_key]}"
            )

        final_dict[key] = structured_data
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


def _structure_data_using_relative_positions(
    data_tmpl: Any,
    data_values_type: str,
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
            _structure_data_using_relative_positions(
                elem, data_values_type, flat_values, allowed_types
            )
            for elem in data_tmpl
        ]

    if type(data_tmpl) not in allowed_types:
        raise TypeError(f"Types allowed: {allowed_types}")

    if data_values_type == smpc_int_type:
        return int(flat_values[data_tmpl])

    return flat_values[data_tmpl]

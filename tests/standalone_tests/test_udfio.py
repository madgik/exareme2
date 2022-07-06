import numpy as np
import pytest

from mipengine.udfgen.udfio import construct_secure_transfer_dict
from mipengine.udfgen.udfio import merge_tensor_to_list
from mipengine.udfgen.udfio import secure_transfers_to_merged_dict
from mipengine.udfgen.udfio import split_secure_transfer_dict


def test_merge_tensor_to_list_2tables_0D():
    columns = dict(
        prefix_node_id=np.array(["a", "b"]),
        prefix_dim0=np.array([0, 0]),
        prefix_val=np.array([1, 2]),
    )
    xs = merge_tensor_to_list(columns)
    assert xs == [np.array([1]), np.array([2])]


def test_merge_tensor_to_list_2tables_1D():
    columns = dict(
        prefix_node_id=np.array(["a", "a", "b", "b"]),
        prefix_dim0=np.array([0, 1, 0, 1]),
        prefix_val=np.array([1, 1, 2, 2]),
    )
    expected_xs = [np.array([1, 1]), np.array([2, 2])]
    xs = merge_tensor_to_list(columns)
    assert all((x == expected_x).all() for x, expected_x in zip(xs, expected_xs))


def test_merge_tensor_to_list_2tables_2D():
    columns = dict(
        prefix_node_id=np.array(["a", "a", "a", "a", "b", "b", "b", "b"]),
        prefix_dim0=np.array([0, 0, 1, 1, 0, 0, 1, 1]),
        prefix_dim1=np.array([0, 1, 0, 1, 0, 1, 0, 1]),
        prefix_val=np.array([1, 1, 1, 1, 2, 2, 2, 2]),
    )
    expected_xs = [np.array([[1, 1], [1, 1]]), np.array([[2, 2], [2, 2]])]
    xs = merge_tensor_to_list(columns)
    assert all((x == expected_x).all() for x, expected_x in zip(xs, expected_xs))


def test_merge_tensor_to_list_3tables_2D():
    columns = dict(
        prefix_node_id=np.array(
            ["a", "a", "a", "a", "b", "b", "b", "b", "c", "c", "c", "c"]
        ),
        prefix_dim0=np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]),
        prefix_dim1=np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
        prefix_val=np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]),
    )
    expected_xs = [
        np.array([[1, 1], [1, 1]]),
        np.array([[2, 2], [2, 2]]),
        np.array([[3, 3], [3, 3]]),
    ]
    xs = merge_tensor_to_list(columns)
    assert all((x == expected_x).all() for x, expected_x in zip(xs, expected_xs))


def test_merge_tensor_to_list_no_nodeid():
    columns = dict(
        prefix_nodeid=np.array(
            ["a", "a", "a", "a", "b", "b", "b", "b", "c", "c", "c", "c"]
        ),
        prefix_dim0=np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]),
        prefix_dim1=np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
        prefix_val=np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]),
    )
    with pytest.raises(ValueError):
        xs = merge_tensor_to_list(columns)


def get_secure_transfers_to_merged_dict_success_cases():
    secure_transfers_cases = [
        pytest.param(
            [
                {
                    "a": {"data": 2, "operation": "sum", "type": "int"},
                },
                {
                    "a": {"data": 3, "operation": "sum", "type": "int"},
                },
            ],
            {"a": 5},
            id="sum operation with ints",
        ),
        pytest.param(
            [
                {
                    "a": {"data": 2.5, "operation": "sum", "type": "float"},
                },
                {
                    "a": {"data": 3.6, "operation": "sum", "type": "float"},
                },
            ],
            {"a": 6.1},
            id="sum operation with floats",
        ),
        pytest.param(
            [
                {
                    "a": {"data": 2, "operation": "sum", "type": "int"},
                    "b": {"data": 5, "operation": "sum", "type": "int"},
                    "c": {"data": 5.123, "operation": "sum", "type": "float"},
                },
                {
                    "a": {"data": 3, "operation": "sum", "type": "int"},
                    "b": {"data": 7, "operation": "sum", "type": "int"},
                    "c": {"data": 5.456, "operation": "sum", "type": "float"},
                },
            ],
            {"a": 5, "b": 12, "c": 10.579},
            id="multiple sum operations with ints/floats",
        ),
        pytest.param(
            [
                {
                    "a": {"data": [1, 2, 3], "operation": "sum", "type": "int"},
                },
                {
                    "a": {"data": [9, 8, 7], "operation": "sum", "type": "int"},
                },
            ],
            {
                "a": [10, 10, 10],
            },
            id="sum operation with list of ints",
        ),
        pytest.param(
            [
                {
                    "a": {"data": 10, "operation": "sum", "type": "int"},
                    "b": {
                        "data": [10, 20, 30, 40, 50, 60],
                        "operation": "sum",
                        "type": "int",
                    },
                    "c": {
                        "data": [[10, 20, 30, 40, 50, 60], [70, 80, 90]],
                        "operation": "sum",
                        "type": "int",
                    },
                },
                {
                    "a": {"data": 100, "operation": "sum", "type": "int"},
                    "b": {
                        "data": [100, 200, 300, 400, 500, 600],
                        "operation": "sum",
                        "type": "int",
                    },
                    "c": {
                        "data": [[100, 200, 300, 400, 500, 600], [700, 800, 900]],
                        "operation": "sum",
                        "type": "int",
                    },
                },
            ],
            {
                "a": 110,
                "b": [110, 220, 330, 440, 550, 660],
                "c": [[110, 220, 330, 440, 550, 660], [770, 880, 990]],
            },
            id="complex sum operations with nested lists",
        ),
        pytest.param(
            [
                {
                    "sum": {"data": 10, "operation": "sum", "type": "int"},
                    "min": {
                        "data": [10, 200, 30, 400, 50, 600],
                        "operation": "min",
                        "type": "int",
                    },
                    "max": {
                        "data": [[100, 20, 300, 40, 500, 60], [700, 80, 900]],
                        "operation": "max",
                        "type": "int",
                    },
                },
                {
                    "sum": {"data": 100, "operation": "sum", "type": "int"},
                    "min": {
                        "data": [100, 20, 300, 40, 500, 60],
                        "operation": "min",
                        "type": "int",
                    },
                    "max": {
                        "data": [[10, 200, 30, 400, 50, 600], [70, 800, 90]],
                        "operation": "max",
                        "type": "int",
                    },
                },
            ],
            {
                "sum": 110,
                "min": [10, 20, 30, 40, 50, 60],
                "max": [[100, 200, 300, 400, 500, 600], [700, 800, 900]],
            },
            id="mixed sum/min/max operations with nested lists",
        ),
        pytest.param(
            [
                {
                    "sum": {"data": 10, "operation": "sum", "type": "int"},
                    "sumfloat": {"data": 10.677, "operation": "sum", "type": "float"},
                    "min": {
                        "data": [10, 200, 30, 400, 50, 600],
                        "operation": "min",
                        "type": "int",
                    },
                    "max": {
                        "data": [[100, 20, 300, 40, 500, 60], [700, 80, 900]],
                        "operation": "max",
                        "type": "int",
                    },
                    "maxfloat": {
                        "data": [
                            [100.5, 200.3, 30.4, 400.5678, 50.6, 600.7],
                            [70.8, 800.9, 90.01],
                        ],
                        "operation": "max",
                        "type": "float",
                    },
                },
                {
                    "sum": {"data": 100, "operation": "sum", "type": "int"},
                    "sumfloat": {"data": 0.678, "operation": "sum", "type": "float"},
                    "min": {
                        "data": [100, 20, 300, 40, 500, 60],
                        "operation": "min",
                        "type": "int",
                    },
                    "max": {
                        "data": [[10, 200, 30, 400, 50, 600], [70, 800, 90]],
                        "operation": "max",
                        "type": "int",
                    },
                    "maxfloat": {
                        "data": [
                            [10.1, 200.25, 30.3, 400.4, 50.5, 600.6],
                            [72, 8000.8, 90.9],
                        ],
                        "operation": "max",
                        "type": "float",
                    },
                },
            ],
            {
                "sum": 110,
                "sumfloat": 11.355,
                "min": [10, 20, 30, 40, 50, 60],
                "max": [[100, 200, 300, 400, 500, 600], [700, 800, 900]],
                "maxfloat": [
                    [100.5, 200.3, 30.4, 400.5678, 50.6, 600.7],
                    [72, 8000.8, 90.9],
                ],
            },
            id="mixed sum/min/max operations with ints/floats and nested lists",
        ),
    ]
    return secure_transfers_cases


@pytest.mark.parametrize(
    "transfers, result", get_secure_transfers_to_merged_dict_success_cases()
)
def test_secure_transfer_to_merged_dict(transfers, result):
    assert secure_transfers_to_merged_dict(transfers) == result


def get_secure_transfers_merged_to_dict_fail_cases():
    secure_transfers_fail_cases = [
        (
            [
                {
                    "a": {"data": 2, "type": "int"},
                },
                {
                    "a": {"data": 3, "operation": "sum", "type": "int"},
                },
            ],
            (
                ValueError,
                "Secure Transfer operation is not provided. Expected format: .*",
            ),
        ),
        (
            [
                {
                    "a": {"data": 2, "operation": "sum", "type": "int"},
                },
                {
                    "a": {"data": 3, "operation": "sum"},
                },
            ],
            (ValueError, "Secure Transfer type is not provided. Expected format: .*"),
        ),
        (
            [
                {
                    "a": {"data": 2, "operation": "sum", "type": "int"},
                },
                {
                    "a": {"data": 3, "operation": "whatever", "type": "int"},
                },
            ],
            (
                ValueError,
                "Secure Transfer operation is not supported: .*",
            ),
        ),
        (
            [
                {
                    "a": {"data": 2, "operation": "sum", "type": "int"},
                },
                {
                    "a": {"data": 3, "operation": "sum", "type": "whatever"},
                },
            ],
            (
                ValueError,
                "Secure Transfer type is not supported: .*",
            ),
        ),
        (
            [
                {
                    "a": {"data": 2, "operation": "sum", "type": "int"},
                },
                {
                    "a": {"data": 3, "operation": "min", "type": "int"},
                },
            ],
            (
                ValueError,
                "Similar secure transfer keys should have the same operation .*",
            ),
        ),
        (
            [
                {
                    "a": {"data": 2, "operation": "sum", "type": "int"},
                },
                {
                    "a": {"data": 3, "operation": "sum", "type": "float"},
                },
            ],
            (
                ValueError,
                "Similar secure transfer keys should have the same type .*",
            ),
        ),
        (
            [
                {
                    "a": {"data": 2, "operation": "sum", "type": "int"},
                },
                {
                    "a": {"data": [3], "operation": "sum", "type": "int"},
                },
            ],
            (ValueError, "Secure transfers' data should have the same structure."),
        ),
        (
            [
                {
                    "a": {"data": "tet", "operation": "sum", "type": "int"},
                },
                {
                    "a": {"data": "tet", "operation": "sum", "type": "int"},
                },
            ],
            (
                TypeError,
                "Secure transfer data must have one of the following types: .*",
            ),
        ),
    ]
    return secure_transfers_fail_cases


@pytest.mark.parametrize(
    "transfers, exception", get_secure_transfers_merged_to_dict_fail_cases()
)
def test_secure_transfers_to_merged_dict_fail_cases(transfers, exception):
    exception_type, exception_message = exception
    with pytest.raises(exception_type, match=exception_message):
        secure_transfers_to_merged_dict(transfers)


def get_secure_transfer_dict_success_cases():
    secure_transfer_cases = [
        pytest.param(
            {
                "a": {"data": 2, "operation": "sum", "type": "int"},
            },
            (
                {
                    "a": {"data": 0, "operation": "sum", "type": "int"},
                },
                [2],
                [],
                [],
            ),
            {
                "a": 2,
            },
            id="sum operation with int",
        ),
        pytest.param(
            {
                "a": {"data": 2.5, "operation": "sum", "type": "float"},
            },
            (
                {
                    "a": {"data": 0, "operation": "sum", "type": "float"},
                },
                [2.5],
                [],
                [],
            ),
            {
                "a": 2.5,
            },
            id="sum operation with float",
        ),
        pytest.param(
            {
                "a": {"data": 2, "operation": "sum", "type": "int"},
                "b": {"data": 5, "operation": "sum", "type": "int"},
            },
            (
                {
                    "a": {"data": 0, "operation": "sum", "type": "int"},
                    "b": {"data": 1, "operation": "sum", "type": "int"},
                },
                [2, 5],
                [],
                [],
            ),
            {"a": 2, "b": 5},
            id="sum operation with ints",
        ),
        pytest.param(
            {
                "a": {"data": 2, "operation": "sum", "type": "int"},
                "b": {"data": 5.5, "operation": "sum", "type": "float"},
            },
            (
                {
                    "a": {"data": 0, "operation": "sum", "type": "int"},
                    "b": {"data": 1, "operation": "sum", "type": "float"},
                },
                [2, 5.5],
                [],
                [],
            ),
            {"a": 2, "b": 5.5},
            id="sum operation with int/float",
        ),
        pytest.param(
            {
                "a": {"data": [1, 2, 3], "operation": "sum", "type": "int"},
            },
            (
                {
                    "a": {"data": [0, 1, 2], "operation": "sum", "type": "int"},
                },
                [1, 2, 3],
                [],
                [],
            ),
            {
                "a": [1, 2, 3],
            },
            id="sum operation with list of ints",
        ),
        pytest.param(
            {
                "a": {"data": 10, "operation": "sum", "type": "int"},
                "b": {
                    "data": [10, 20, 30, 40, 50, 60],
                    "operation": "sum",
                    "type": "int",
                },
                "c": {
                    "data": [[10, 20, 30, 40, 50, 60], [70, 80, 90]],
                    "operation": "sum",
                    "type": "int",
                },
            },
            (
                {
                    "a": {"data": 0, "operation": "sum", "type": "int"},
                    "b": {
                        "data": [1, 2, 3, 4, 5, 6],
                        "operation": "sum",
                        "type": "int",
                    },
                    "c": {
                        "data": [[7, 8, 9, 10, 11, 12], [13, 14, 15]],
                        "operation": "sum",
                        "type": "int",
                    },
                },
                [10, 10, 20, 30, 40, 50, 60, 10, 20, 30, 40, 50, 60, 70, 80, 90],
                [],
                [],
            ),
            {
                "a": 10,
                "b": [10, 20, 30, 40, 50, 60],
                "c": [[10, 20, 30, 40, 50, 60], [70, 80, 90]],
            },
            id="sum operation with nested lists of ints",
        ),
        pytest.param(
            {
                "min": {"data": [2, 5.6], "operation": "min", "type": "float"},
            },
            (
                {
                    "min": {"data": [0, 1], "operation": "min", "type": "float"},
                },
                [],
                [2, 5.6],
                [],
            ),
            {
                "min": [2, 5.6],
            },
            id="min operation with int/float",
        ),
        pytest.param(
            {
                "max": {"data": [2, 5.6], "operation": "max", "type": "float"},
            },
            (
                {
                    "max": {"data": [0, 1], "operation": "max", "type": "float"},
                },
                [],
                [],
                [2, 5.6],
            ),
            {
                "max": [2, 5.6],
            },
            id="max operation with int/float",
        ),
        pytest.param(
            {
                "sum1": {"data": [1, 2, 3, 4.5], "operation": "sum", "type": "float"},
                "sum2": {"data": [6, 7.8], "operation": "sum", "type": "float"},
                "min1": {"data": [6, 7.8], "operation": "min", "type": "float"},
                "min2": {"data": [1.5, 2.0], "operation": "min", "type": "float"},
                "max1": {"data": [6.8, 7], "operation": "max", "type": "float"},
                "max2": {"data": [1.5, 2], "operation": "max", "type": "float"},
            },
            (
                {
                    "sum1": {"data": [0, 1, 2, 3], "operation": "sum", "type": "float"},
                    "sum2": {"data": [4, 5], "operation": "sum", "type": "float"},
                    "min1": {"data": [0, 1], "operation": "min", "type": "float"},
                    "min2": {"data": [2, 3], "operation": "min", "type": "float"},
                    "max1": {"data": [0, 1], "operation": "max", "type": "float"},
                    "max2": {"data": [2, 3], "operation": "max", "type": "float"},
                },
                [1, 2, 3, 4.5, 6, 7.8],
                [6, 7.8, 1.5, 2.0],
                [6.8, 7, 1.5, 2],
            ),
            {
                "sum1": [1, 2, 3, 4.5],
                "sum2": [6, 7.8],
                "min1": [6, 7.8],
                "min2": [1.5, 2.0],
                "max1": [6.8, 7],
                "max2": [1.5, 2],
            },
            id="mixed sum/min/max operation with mixed ints/floats",
        ),
        pytest.param(
            {
                "sum": {"data": [100, 200, 300], "operation": "sum", "type": "int"},
                "sumfloat": {
                    "data": [1.2, 2.3, 3.4],
                    "operation": "sum",
                    "type": "float",
                },
                "max": {"data": 58, "operation": "max", "type": "int"},
            },
            (
                {
                    "sum": {"data": [0, 1, 2], "operation": "sum", "type": "int"},
                    "sumfloat": {
                        "data": [3, 4, 5],
                        "operation": "sum",
                        "type": "float",
                    },
                    "max": {"data": 0, "operation": "max", "type": "int"},
                },
                [100, 200, 300, 1.2, 2.3, 3.4],
                [],
                [58],
            ),
            {
                "sum": [100, 200, 300],
                "sumfloat": [1.2, 2.3, 3.4],
                "max": 58,
            },
            id="sum operations with ints/floats in separate keys",
        ),
    ]
    return secure_transfer_cases


@pytest.mark.parametrize(
    "secure_transfer, smpc_parts, final_result",
    get_secure_transfer_dict_success_cases(),
)
def test_split_secure_transfer_dict(secure_transfer, smpc_parts, final_result):
    assert split_secure_transfer_dict(secure_transfer) == smpc_parts


def get_split_secure_transfer_dict_fail_cases():
    split_secure_transfer_dict_fail_cases = [
        (
            {
                "a": {"data": 3, "operation": "whatever", "type": "int"},
            },
            (
                ValueError,
                "Secure Transfer operation is not supported: .*",
            ),
        ),
        (
            {
                "a": {"data": 3, "operation": "sum", "type": "whatever"},
            },
            (
                ValueError,
                "Secure Transfer type is not supported: .*",
            ),
        ),
        (
            {
                "a": {"data": "tet", "operation": "sum", "type": "int"},
            },
            (
                TypeError,
                "Secure Transfer key: 'a', operation: 'sum'. Error: Types allowed: .*",
            ),
        ),
        (
            {
                "a": {"llalal": 0, "operation": "sum", "type": "int"},
            },
            (
                ValueError,
                "Each Secure Transfer key should contain data.",
            ),
        ),
        (
            {
                "a": {"data": 0, "sdfs": "sum"},
            },
            (
                ValueError,
                "Each Secure Transfer key should contain an operation.",
            ),
        ),
        (
            {
                "a": {"data": 0, "operation": "sum"},
            },
            (
                ValueError,
                "Each Secure Transfer key should contain a type.",
            ),
        ),
    ]
    return split_secure_transfer_dict_fail_cases


@pytest.mark.parametrize(
    "result, exception", get_split_secure_transfer_dict_fail_cases()
)
def test_split_secure_transfer_dict_fail_cases(result, exception):
    exception_type, exception_message = exception
    with pytest.raises(exception_type, match=exception_message):
        split_secure_transfer_dict(result)


@pytest.mark.parametrize(
    "secure_transfer, smpc_parts, final_result",
    get_secure_transfer_dict_success_cases(),
)
def test_construct_secure_transfer_dict(secure_transfer, smpc_parts, final_result):
    assert construct_secure_transfer_dict(*smpc_parts) == final_result


def test_proper_int_casting_in_construct_secure_transfer_dict():
    """
    SMPC will only return floats, so we need to make sure that the final values will
    be converted to the proper int type, if an int type is provided.
    """
    input_values = (
        {
            "sum": {"data": [0, 1, 2], "operation": "sum", "type": "int"},
            "min": {"data": [0, 1, 2], "operation": "sum", "type": "int"},
            "max": {"data": 0, "operation": "max", "type": "int"},
        },
        [100.0, 200.0, 300.0],
        [10.0, 20.0, 30.0],
        [1.0],
    )

    assert construct_secure_transfer_dict(*input_values) == {
        "sum": [100, 200, 300],
        "min": [100, 200, 300],
        "max": 1,
    }

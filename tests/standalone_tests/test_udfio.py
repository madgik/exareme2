import pytest
import numpy as np

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
        (
            [
                {
                    "a": {"data": 2, "operation": "sum"},
                },
                {
                    "a": {"data": 3, "operation": "sum"},
                },
            ],
            {"a": 5},
        ),
        (
            [
                {
                    "a": {"data": 2, "operation": "sum"},
                    "b": {"data": 5, "operation": "sum"},
                },
                {
                    "a": {"data": 3, "operation": "sum"},
                    "b": {"data": 7, "operation": "sum"},
                },
            ],
            {"a": 5, "b": 12},
        ),
        (
            [
                {
                    "a": {"data": [1, 2, 3], "operation": "sum"},
                },
                {
                    "a": {"data": [9, 8, 7], "operation": "sum"},
                },
            ],
            {
                "a": [10, 10, 10],
            },
        ),
        (
            [
                {
                    "a": {"data": 10, "operation": "sum"},
                    "b": {
                        "data": [10, 20, 30, 40, 50, 60],
                        "operation": "sum",
                    },
                    "c": {
                        "data": [[10, 20, 30, 40, 50, 60], [70, 80, 90]],
                        "operation": "sum",
                    },
                },
                {
                    "a": {"data": 100, "operation": "sum"},
                    "b": {
                        "data": [100, 200, 300, 400, 500, 600],
                        "operation": "sum",
                    },
                    "c": {
                        "data": [[100, 200, 300, 400, 500, 600], [700, 800, 900]],
                        "operation": "sum",
                    },
                },
            ],
            {
                "a": 110,
                "b": [110, 220, 330, 440, 550, 660],
                "c": [[110, 220, 330, 440, 550, 660], [770, 880, 990]],
            },
        ),
        (
            [
                {
                    "sum": {"data": 10, "operation": "sum"},
                    "min": {
                        "data": [10, 200, 30, 400, 50, 600],
                        "operation": "min",
                    },
                    "max": {
                        "data": [[100, 20, 300, 40, 500, 60], [700, 80, 900]],
                        "operation": "max",
                    },
                },
                {
                    "sum": {"data": 100, "operation": "sum"},
                    "min": {
                        "data": [100, 20, 300, 40, 500, 60],
                        "operation": "min",
                    },
                    "max": {
                        "data": [[10, 200, 30, 400, 50, 600], [70, 800, 90]],
                        "operation": "max",
                    },
                },
            ],
            {
                "sum": 110,
                "min": [10, 20, 30, 40, 50, 60],
                "max": [[100, 200, 300, 400, 500, 600], [700, 800, 900]],
            },
        ),
    ]
    return secure_transfers_cases


@pytest.mark.parametrize(
    "transfers, result", get_secure_transfers_to_merged_dict_success_cases()
)
def test_secure_transfer_to_merged_dict(transfers, result):
    assert secure_transfers_to_merged_dict(transfers) == result


def get_secure_transfer_dict_success_cases():
    secure_transfer_cases = [
        (
            {
                "a": {"data": 2, "operation": "sum"},
            },
            (
                {
                    "a": {"data": 0, "operation": "sum"},
                },
                [2],
                [],
                [],
                [],
            ),
            {
                "a": 2,
            },
        ),
        (
            {
                "a": {"data": 2, "operation": "sum"},
                "b": {"data": 5, "operation": "sum"},
            },
            (
                {
                    "a": {"data": 0, "operation": "sum"},
                    "b": {"data": 1, "operation": "sum"},
                },
                [2, 5],
                [],
                [],
                [],
            ),
            {"a": 2, "b": 5},
        ),
        (
            {
                "a": {"data": [1, 2, 3], "operation": "sum"},
            },
            (
                {
                    "a": {"data": [0, 1, 2], "operation": "sum"},
                },
                [1, 2, 3],
                [],
                [],
                [],
            ),
            {
                "a": [1, 2, 3],
            },
        ),
        (
            {
                "a": {"data": 10, "operation": "sum"},
                "b": {
                    "data": [10, 20, 30, 40, 50, 60],
                    "operation": "sum",
                },
                "c": {
                    "data": [[10, 20, 30, 40, 50, 60], [70, 80, 90]],
                    "operation": "sum",
                },
            },
            (
                {
                    "a": {"data": 0, "operation": "sum"},
                    "b": {
                        "data": [1, 2, 3, 4, 5, 6],
                        "operation": "sum",
                    },
                    "c": {
                        "data": [[7, 8, 9, 10, 11, 12], [13, 14, 15]],
                        "operation": "sum",
                    },
                },
                [10, 10, 20, 30, 40, 50, 60, 10, 20, 30, 40, 50, 60, 70, 80, 90],
                [],
                [],
                [],
            ),
            {
                "a": 10,
                "b": [10, 20, 30, 40, 50, 60],
                "c": [[10, 20, 30, 40, 50, 60], [70, 80, 90]],
            },
        ),
        (
            {
                "min": {"data": [2, 5.6], "operation": "min"},
            },
            (
                {
                    "min": {"data": [0, 1], "operation": "min"},
                },
                [],
                [2, 5.6],
                [],
                [],
            ),
            {
                "min": [2, 5.6],
            },
        ),
        (
            {
                "max": {"data": [2, 5.6], "operation": "max"},
            },
            (
                {
                    "max": {"data": [0, 1], "operation": "max"},
                },
                [],
                [],
                [2, 5.6],
                [],
            ),
            {
                "max": [2, 5.6],
            },
        ),
        (
            {
                "sum1": {"data": [1, 2, 3, 4.5], "operation": "sum"},
                "sum2": {"data": [6, 7.8], "operation": "sum"},
                "min1": {"data": [6, 7.8], "operation": "min"},
                "min2": {"data": [1.5, 2.0], "operation": "min"},
                "max1": {"data": [6.8, 7], "operation": "max"},
                "max2": {"data": [1.5, 2], "operation": "max"},
            },
            (
                {
                    "sum1": {"data": [0, 1, 2, 3], "operation": "sum"},
                    "sum2": {"data": [4, 5], "operation": "sum"},
                    "min1": {"data": [0, 1], "operation": "min"},
                    "min2": {"data": [2, 3], "operation": "min"},
                    "max1": {"data": [0, 1], "operation": "max"},
                    "max2": {"data": [2, 3], "operation": "max"},
                },
                [1, 2, 3, 4.5, 6, 7.8],
                [6, 7.8, 1.5, 2.0],
                [6.8, 7, 1.5, 2],
                [],
            ),
            {
                "sum1": [1, 2, 3, 4.5],
                "sum2": [6, 7.8],
                "min1": [6, 7.8],
                "min2": [1.5, 2.0],
                "max1": [6.8, 7],
                "max2": [1.5, 2],
            },
        ),
        (
            {
                "sum": {"data": [100, 200, 300], "operation": "sum"},
                "max": {"data": 58, "operation": "max"},
            },
            (
                {
                    "sum": {"data": [0, 1, 2], "operation": "sum"},
                    "max": {"data": 0, "operation": "max"},
                },
                [100, 200, 300],
                [],
                [58],
                [],
            ),
            {
                "sum": [100, 200, 300],
                "max": 58,
            },
        ),
    ]
    return secure_transfer_cases


@pytest.mark.parametrize(
    "secure_transfer, smpc_parts, final_result",
    get_secure_transfer_dict_success_cases(),
)
def test_split_secure_transfer_dict(secure_transfer, smpc_parts, final_result):
    assert split_secure_transfer_dict(secure_transfer) == smpc_parts


@pytest.mark.parametrize(
    "secure_transfer, smpc_parts, final_result",
    get_secure_transfer_dict_success_cases(),
)
def test_construct_secure_transfer_dict(secure_transfer, smpc_parts, final_result):
    assert construct_secure_transfer_dict(*smpc_parts) == final_result


def get_secure_transfers_merged_to_dict_fail_cases():
    secure_transfers_fail_cases = [
        (
            [
                {
                    "a": {"data": 2, "operation": "sum"},
                },
                {
                    "a": {"data": 3, "operation": "whatever"},
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
                    "a": {"data": 2, "operation": "sum"},
                },
                {
                    "a": {"data": 3, "operation": "min"},
                },
            ],
            (
                ValueError,
                "All secure transfer keys should have the same 'operation' .*",
            ),
        ),
        (
            [
                {
                    "a": {"data": 2, "operation": "sum"},
                },
                {
                    "a": {"data": [3], "operation": "sum"},
                },
            ],
            (ValueError, "Secure transfers' data should have the same structure."),
        ),
        (
            [
                {
                    "a": {"data": "tet", "operation": "sum"},
                },
                {
                    "a": {"data": "tet", "operation": "sum"},
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


def get_split_secure_transfer_dict_fail_cases():
    split_secure_transfer_dict_fail_cases = [
        (
            {
                "a": {"data": 3, "operation": "whatever"},
            },
            (
                ValueError,
                "Secure Transfer operation is not supported: .*",
            ),
        ),
        (
            {
                "a": {"data": "tet", "operation": "sum"},
            },
            (
                TypeError,
                "Secure Transfer key: 'a', operation: 'sum'. Error: Types allowed: .*",
            ),
        ),
        (
            {
                "a": {"llalal": 0, "operation": "sum"},
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
    ]
    return split_secure_transfer_dict_fail_cases


@pytest.mark.parametrize(
    "result, exception", get_split_secure_transfer_dict_fail_cases()
)
def test_split_secure_transfer_dict_fail_cases(result, exception):
    exception_type, exception_message = exception
    with pytest.raises(exception_type, match=exception_message):
        split_secure_transfer_dict(result)

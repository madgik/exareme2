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
                    "a": {"data": 2, "type": "int", "operation": "addition"},
                },
                {
                    "a": {"data": 3, "type": "int", "operation": "addition"},
                },
            ],
            {"a": 5},
        ),
        (
            [
                {
                    "a": {"data": 2, "type": "int", "operation": "addition"},
                    "b": {"data": 5, "type": "int", "operation": "addition"},
                },
                {
                    "a": {"data": 3, "type": "int", "operation": "addition"},
                    "b": {"data": 7, "type": "int", "operation": "addition"},
                },
            ],
            {"a": 5, "b": 12},
        ),
        (
            [
                {
                    "a": {"data": [1, 2, 3], "type": "int", "operation": "addition"},
                },
                {
                    "a": {"data": [9, 8, 7], "type": "int", "operation": "addition"},
                },
            ],
            {
                "a": [10, 10, 10],
            },
        ),
        (
            [
                {
                    "a": {"data": 10, "type": "int", "operation": "addition"},
                    "b": {
                        "data": [10, 20, 30, 40, 50, 60],
                        "type": "int",
                        "operation": "addition",
                    },
                    "c": {
                        "data": [[10, 20, 30, 40, 50, 60], [70, 80, 90]],
                        "type": "int",
                        "operation": "addition",
                    },
                },
                {
                    "a": {"data": 100, "type": "int", "operation": "addition"},
                    "b": {
                        "data": [100, 200, 300, 400, 500, 600],
                        "type": "int",
                        "operation": "addition",
                    },
                    "c": {
                        "data": [[100, 200, 300, 400, 500, 600], [700, 800, 900]],
                        "type": "int",
                        "operation": "addition",
                    },
                },
            ],
            {
                "a": 110,
                "b": [110, 220, 330, 440, 550, 660],
                "c": [[110, 220, 330, 440, 550, 660], [770, 880, 990]],
            },
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
                    "a": {"data": 2, "type": "int", "operation": "addition"},
                },
                {
                    "a": {"data": 3, "type": "int", "operation": "whatever"},
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
                    "a": {"data": 2, "type": "int", "operation": "addition"},
                },
                {
                    "a": {"data": 3, "type": "decimal", "operation": "addition"},
                },
            ],
            (ValueError, "All secure transfer keys should have the same 'type' .*"),
        ),
        (
            [
                {
                    "a": {"data": 2, "type": "int", "operation": "addition"},
                },
                {
                    "a": {"data": [3], "type": "int", "operation": "addition"},
                },
            ],
            (TypeError, "Secure transfer data have different types: .*"),
        ),
        (
            [
                {
                    "a": {"data": 2, "type": "whatever", "operation": "addition"},
                },
                {
                    "a": {"data": 3, "type": "whatever", "operation": "addition"},
                },
            ],
            (
                NotImplementedError,
                "Secure transfer type: .* not supported for operation: .*",
            ),
        ),
        (
            [
                {
                    "a": {"data": 2, "type": "int", "operation": "whatever"},
                },
                {
                    "a": {"data": 3, "type": "int", "operation": "addition"},
                },
            ],
            (NotImplementedError, "Secure transfer operation not supported: .*"),
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
        (
            {
                "a": {"data": 2, "type": "int", "operation": "addition"},
            },
            (
                {
                    "a": {"data": 0, "type": "int", "operation": "addition"},
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
                "a": {"data": 2, "type": "int", "operation": "addition"},
                "b": {"data": 5, "type": "int", "operation": "addition"},
            },
            (
                {
                    "a": {"data": 0, "type": "int", "operation": "addition"},
                    "b": {"data": 1, "type": "int", "operation": "addition"},
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
                "a": {"data": [1, 2, 3], "type": "int", "operation": "addition"},
            },
            (
                {
                    "a": {"data": [0, 1, 2], "type": "int", "operation": "addition"},
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
                "a": {"data": 10, "type": "int", "operation": "addition"},
                "b": {
                    "data": [10, 20, 30, 40, 50, 60],
                    "type": "int",
                    "operation": "addition",
                },
                "c": {
                    "data": [[10, 20, 30, 40, 50, 60], [70, 80, 90]],
                    "type": "int",
                    "operation": "addition",
                },
            },
            (
                {
                    "a": {"data": 0, "type": "int", "operation": "addition"},
                    "b": {
                        "data": [1, 2, 3, 4, 5, 6],
                        "type": "int",
                        "operation": "addition",
                    },
                    "c": {
                        "data": [[7, 8, 9, 10, 11, 12], [13, 14, 15]],
                        "type": "int",
                        "operation": "addition",
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

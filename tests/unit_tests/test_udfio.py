import pytest
import numpy as np

from mipengine.node.udfgen.udfio import merge_tensor_to_list


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

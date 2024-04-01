import pytest

from exareme2.algorithms.exareme2.fedaverage import fed_average


def test_fed_average__scalar():
    result = fed_average({"a": 1}, num_local_workers=2)
    assert result == {"a": 0.5}


def test_fed_average__1d():
    result = fed_average({"a": [1, 2]}, num_local_workers=2)
    assert result == {"a": [0.5, 1.0]}


def test_fed_average__2d():
    result = fed_average({"a": [[1, 2], [3, 4]]}, num_local_workers=2)
    assert result == {"a": [[0.5, 1.0], [1.5, 2.0]]}


def test_fed_average__mixed():
    params = {"a": 1, "b": [1, 2], "c": [[1, 2], [3, 4]]}
    result = fed_average(params, num_local_workers=2)
    assert result == {"a": 0.5, "b": [0.5, 1.0], "c": [[0.5, 1.0], [1.5, 2.0]]}


def test_fed_average__value_error():
    params = {"a": [[1, 2], [3]]}
    # params values should be list representation of n-dimensional arrays
    with pytest.raises(ValueError):
        fed_average(params, num_local_workers=1)

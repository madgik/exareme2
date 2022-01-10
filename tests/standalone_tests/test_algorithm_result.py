import pytest

from mipengine.algorithm_flow_DTOs import TabularDataResult


@pytest.fixture
def columns():
    return [
        {"name": "a", "type": "number"},
        {"name": "b", "type": "number"},
        {"name": "c", "type": "number"},
    ]


def test_tabular_data_result(columns):
    result = TabularDataResult(
        title="The Title", columns=columns, data=[[1, 2, 3], [10, 20, 30]]
    )
    assert result.dict() == {
        "title": "The Title",
        "columns": columns,
        "data": [[1, 2, 3], [10, 20, 30]],
    }


def test_tabular_data_result_floats(columns):
    result = TabularDataResult(
        title="The Title", columns=columns, data=[[1.1, 2.2, 3.3], [10, 20, 30]]
    )
    assert result.dict() == {
        "title": "The Title",
        "columns": columns,
        "data": [[1.1, 2.2, 3.3], [10, 20, 30]],
    }


def test_tabular_data_result_to_json(columns):
    result = TabularDataResult(
        title="The Title", columns=columns, data=[[1, 2, 3], [10, 20, 30]]
    )
    expected = '{"title": "The Title", "columns": [{"name": "a", "type": "number"}, {"name": "b", "type": "number"}, {"name": "c", "type": "number"}], "data": [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]}'
    assert result.json() == expected

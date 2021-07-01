from mipengine.algorithms.result import TabularDataResult


def test_tabular_data_result():
    result = TabularDataResult(
        title="The Title", columns=["a", "b", "c"], data=[[1, 2, 3], [10, 20, 30]]
    )
    assert result.dict() == {
        "title": "The Title",
        "columns": ["a", "b", "c"],
        "data": [[1, 2, 3], [10, 20, 30]],
    }


def test_tabular_data_result_floats():
    result = TabularDataResult(
        title="The Title", columns=["a", "b", "c"], data=[[1.1, 2.2, 3.3], [10, 20, 30]]
    )
    assert result.dict() == {
        "title": "The Title",
        "columns": ["a", "b", "c"],
        "data": [[1.1, 2.2, 3.3], [10, 20, 30]],
    }


def test_tabular_data_result_to_json():
    result = TabularDataResult(
        title="The Title", columns=["a", "b", "c"], data=[[1, 2, 3], [10, 20, 30]]
    )
    assert (
        result.json()
        == '{"title": "The Title", "columns": ["a", "b", "c"], "data": [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]}'
    )

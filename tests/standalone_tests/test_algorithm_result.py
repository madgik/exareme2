from mipengine import DType
from mipengine.table_data_DTOs import ColumnDataFloat
from mipengine.table_data_DTOs import ColumnDataInt
from mipengine.table_data_DTOs import ColumnDataStr
from mipengine.algorithm_result_DTOs import TabularDataResult


def test_tabular_data_result():

    columns = [
        ColumnDataFloat(name="float", data=[1.0, 10.0]),
        ColumnDataStr(name="str", data=["2", "20"]),
        ColumnDataInt(name="int", data=[3, 30]),
    ]
    result = TabularDataResult(title="The Title", columns=columns)
    expected_dict = {
        "title": "The Title",
        "columns": [
            {"name": "float", "type": DType.FLOAT, "data": [1.0, 10.0]},
            {"name": "str", "type": DType.STR, "data": ["2", "20"]},
            {"name": "int", "type": DType.INT, "data": [3, 30]},
        ],
    }

    assert result.dict() == expected_dict


def test_tabular_data_result_to_json():
    columns = [
        ColumnDataFloat(name="float", data=[1.0, 10.0]),
        ColumnDataStr(name="str", data=["2", "20"]),
        ColumnDataInt(name="int", data=[3, 30]),
    ]
    result = TabularDataResult(title="The Title", columns=columns)
    expected = '{"title": "The Title", "columns": [{"name": "float", "type": "FLOAT", "data": [1.0, 10.0]}, {"name": "str", "type": "STR", "data": ["2", "20"]}, {"name": "int", "type": "INT", "data": [3, 30]}]}'
    assert result.json() == expected

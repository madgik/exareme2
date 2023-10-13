from exareme2 import DType
from exareme2.node_communication import ColumnDataFloat
from exareme2.node_communication import ColumnDataInt
from exareme2.node_communication import ColumnDataStr
from exareme2.node_communication import TabularDataResult


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
    assert TabularDataResult.parse_raw(result.json()) == result

from typing import List

import pytest
from pydantic import ValidationError

from mipengine.node_tasks_DTOs import (
    DType,
    ColumnInfo,
    TableSchema,
    TableInfo,
    TableView,
    TableData,
    NodeUDFDTO,
)
from mipengine.node_tasks_DTOs import NodeLiteralDTO
from mipengine.node_tasks_DTOs import NodeSMPCDTO
from mipengine.node_tasks_DTOs import NodeSMPCValueDTO
from mipengine.node_tasks_DTOs import NodeSMPCDTO
from mipengine.node_tasks_DTOs import NodeSMPCValueDTO
from mipengine.node_tasks_DTOs import TableType
from mipengine.node_tasks_DTOs import NodeTableDTO
from mipengine.node_tasks_DTOs import NodeTableDTO
from mipengine.node_tasks_DTOs import UDFResults
from mipengine.node_tasks_DTOs import _NodeUDFDTOType
from mipengine.node_tasks_DTOs import UDFKeyArguments
from mipengine.node_tasks_DTOs import UDFPosArguments
from mipengine.node_tasks_DTOs import NodeUDFDTO
from mipengine.node_tasks_DTOs import _NodeUDFDTOType


@pytest.fixture
def acceptable_name():
    return "AcceptableName"


@pytest.fixture
def name_error_str():
    return "Unacceptable Name"


@pytest.fixture
def column_info_data_type():
    return DType.INT


@pytest.fixture
def column_info_data_type_error():
    return 3.4j


# asserts correct parameters in test_column_info
def test_column_info(acceptable_name, column_info_data_type):
    expected_name = "AcceptableName"
    expected_dt = column_info_data_type
    assert expected_name == acceptable_name
    assert expected_dt == column_info_data_type


# raises error for wrong params in ColumnInfo
# @pytest.mark.xfail
def test_column_info_error(name_error_str, column_info_data_type_error):
    with pytest.raises(ValidationError) as exception:
        ColumnInfo(name=name_error_str, dtype=column_info_data_type_error)


def test_column_info_immutable():
    info = ColumnInfo(name="name", dtype=DType.FLOAT)
    with pytest.raises(TypeError):
        info.name = "newname"


# asserts correct parameters in test_column_info
def test_table_schema():
    cols = [
        ColumnInfo(name="AcceptableName1", dtype=DType.FLOAT),
        ColumnInfo(name="AcceptableName2", dtype=DType.FLOAT),
    ]
    assert isinstance(cols, List)
    assert isinstance(cols[0], ColumnInfo)
    assert isinstance(cols[1], ColumnInfo)


# validation check for TableSchema with error
def test_table_schema_sql_injection_error():
    with pytest.raises(ValidationError):
        ColumnInfo(name="Robert'); DROP TABLE data; --", dtype=DType.FLOAT)


def test_table_schema_type_error():
    with pytest.raises(ValidationError):
        ColumnInfo(name=123, dtype=DType.FLOAT)


def test_table_schema_immutable():
    schema = TableSchema(
        columns=[
            ColumnInfo(name="layla", dtype=DType.FLOAT),
            ColumnInfo(name="sheila", dtype=DType.FLOAT),
        ]
    )
    with pytest.raises(TypeError):
        schema.columns = [
            ColumnInfo(name="newname", dtype=DType.FLOAT),
            ColumnInfo(name="newname", dtype=DType.FLOAT),
        ]


@pytest.fixture
def table_info_proper_type():
    return TableInfo(
        name="test",
        schema_=TableSchema(
            columns=[
                ColumnInfo(name="layla", dtype=DType.FLOAT),
                ColumnInfo(name="sheila", dtype=DType.FLOAT),
            ]
        ),
        type_=TableType.NORMAL,
    )


def test_table_info_type(table_info_proper_type):
    assert isinstance(table_info_proper_type.type_, TableType)


@pytest.fixture
def table_info_data_schema():
    return TableSchema(
        columns=[
            ColumnInfo(name="layla", dtype=DType.FLOAT),
            ColumnInfo(name="sheila", dtype=DType.FLOAT),
        ]
    )


def test_table_info_schema(table_info_data_schema):
    assert isinstance(table_info_data_schema, TableSchema)


@pytest.fixture
def table_info_data_schema_error():
    return TableSchema(
        columns=[
            ColumnInfo(name=name_error_str, dtype=DType.FLOAT),
            ColumnInfo(name="Sheila", dtype=DType.FLOAT),
        ]
    )


# validation check for table_info
def test_table_info_error():
    with pytest.raises(ValidationError):
        TableInfo(
            name=name_error_str,
            schema_=table_info_data_schema_error,
            type_=TableType.NORMAL,
        )


def test_table_info_immutable():
    info = TableInfo(
        name="name",
        schema_=TableSchema(
            columns=[
                ColumnInfo(name="layla", dtype=DType.FLOAT),
                ColumnInfo(name="sheila", dtype=DType.FLOAT),
            ],
        ),
        type_=TableType.NORMAL,
    )
    with pytest.raises(TypeError):
        info.name = "newname"


def test_table_view_error():
    with pytest.raises(ValidationError):
        TableView(datasets=[34, "bar", "baz"], columns=[], filter=[])


def test_table_view_immutable():
    view = TableView(
        datasets=["name"],
        columns=["test"],
        filter={},
    )
    with pytest.raises(TypeError):
        view.datasets = []


def test_table_data_error():
    with pytest.raises(ValidationError):
        TableData(schema_="foo", data_=34)


def test_table_data_immutable():
    data = TableData(
        schema_=TableSchema(
            columns=[
                ColumnInfo(name="layla", dtype=DType.FLOAT),
                ColumnInfo(name="sheila", dtype=DType.FLOAT),
            ],
        ),
        data_=[],
    )
    with pytest.raises(TypeError):
        data.data_ = []


def test_table_data():
    with pytest.raises(ValidationError):
        TableData(
            schema_="this is not a TableSchema object",
            data_="and this is not a list of tuples",
        )


def test_udf_dto_instantiation():
    with pytest.raises(ValidationError) as exc:
        NodeUDFDTO(type="LITERAL", value="this can be anything")
    assert "should not be instantiated." in str(exc)


def test_udf_dtos_immutable():
    argument = NodeTableDTO(value="whatever")
    with pytest.raises(TypeError) as exc:
        argument.value = "new"
    assert "is immutable" in str(exc)

    argument = NodeLiteralDTO(value=10)
    with pytest.raises(TypeError) as exc:
        argument.value = "new"
    assert "is immutable" in str(exc)

    argument = NodeSMPCDTO(
        value=NodeSMPCValueDTO(template=NodeTableDTO(value="whatever"))
    )
    with pytest.raises(TypeError) as exc:
        argument.value = "new"
    assert "is immutable" in str(exc)


def test_udf_dtos_correct_type():
    argument = NodeTableDTO(value="whatever")
    assert argument.type == _NodeUDFDTOType.TABLE

    argument = NodeLiteralDTO(value=10)
    assert argument.type == _NodeUDFDTOType.LITERAL

    argument = NodeSMPCDTO(
        value=NodeSMPCValueDTO(template=NodeTableDTO(value="whatever"))
    )
    assert argument.type == _NodeUDFDTOType.SMPC


def get_udf_args_cases():
    return [
        [
            NodeTableDTO(value="whatever"),
        ],
        [NodeTableDTO(value="whatever"), NodeLiteralDTO(value="whatever")],
        [
            NodeTableDTO(value="whatever"),
            NodeLiteralDTO(value="whatever"),
            NodeSMPCDTO(
                value=NodeSMPCValueDTO(template=NodeTableDTO(value="whatever"))
            ),
        ],
    ]


@pytest.mark.parametrize("args", get_udf_args_cases())
def test_pos_udf_arguments_correct_resolutions(args):
    pos_args = UDFPosArguments(args=args)
    pos_args_json = pos_args.json()

    pos_args_unpacked = UDFPosArguments.parse_raw(pos_args_json)

    assert pos_args == pos_args_unpacked


@pytest.mark.parametrize("args", get_udf_args_cases())
def test_kw_udf_arguments_correct_resolutions(args):
    kw_args = UDFKeyArguments(args={pos: arg for pos, arg in enumerate(args)})
    kw_args_json = kw_args.json()

    kw_args_unpacked = UDFKeyArguments.parse_raw(kw_args_json)

    assert kw_args == kw_args_unpacked


def get_udf_results_cases():
    return [
        UDFResults(
            results=[
                NodeTableDTO(value="whatever"),
            ],
        ),
        UDFResults(
            results=[
                NodeSMPCDTO(
                    value=NodeSMPCValueDTO(template=NodeTableDTO(value="whatever"))
                ),
            ],
        ),
        UDFResults(
            results=[
                NodeTableDTO(value="whatever"),
                NodeSMPCDTO(
                    value=NodeSMPCValueDTO(template=NodeTableDTO(value="whatever"))
                ),
            ],
        ),
    ]


@pytest.mark.parametrize("udf_results", get_udf_results_cases())
def test_udf_results_correct_resolutions(udf_results):
    udf_results_json = udf_results.json()

    udf_results_unpacked = UDFResults.parse_raw(udf_results_json)

    print(udf_results)
    print(udf_results_unpacked)

    assert udf_results == udf_results_unpacked

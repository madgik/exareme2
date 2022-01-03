from typing import List

import pytest
from pydantic import ValidationError

from mipengine.tabular_data_DTOs import ColumnDataFloat
from mipengine.tabular_data_DTOs import ColumnDataInt
from mipengine.tabular_data_DTOs import ColumnDataStr
from mipengine.node_tasks_DTOs import (
    DType,
    ColumnInfo,
    TableSchema,
    TableInfo,
    TabularData,
    UDFArgument,
)
from mipengine.node_tasks_DTOs import TableType
from mipengine.node_tasks_DTOs import UDFArgumentKind


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


def test_tabular_data_with_different_column_types():
    expected_columns = [
        ColumnDataFloat(data=[1.0, None], name="column1"),
        ColumnDataInt(data=[2, None], name="column2"),
        ColumnDataStr(data=["3", None], name="column3"),
        ColumnDataInt(data=[4, 4], name="column4"),
        ColumnDataFloat(data=[5.0, 5.0], name="column5"),
    ]
    data = TabularData(
        name="table_name",
        columns=expected_columns,
    )
    assert data.columns == expected_columns
    assert TabularData.parse_raw(data.json()) == data


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


def test_tabular_data_error():
    with pytest.raises(ValidationError):
        TabularData(name="foo", columns=34)


def test_tabular_data_immutable():
    data = TabularData(
        name="table",
        columns=[
            ColumnDataFloat(name="layla", data=[9.1]),
            ColumnDataFloat(name="sheila", data=[9.1]),
        ],
    )
    with pytest.raises(TypeError):
        data.name = "newname"


def test_tabular_data():
    with pytest.raises(ValidationError):
        TabularData(
            name="this is not a table name object",
            columns="and this is not a list of columns",
        )


def test_udf_argument():
    with pytest.raises(ValidationError):
        UDFArgument(kind="Not a UDFArgumentKind", value="this can be anything")


def test_udf_argument_immutable():
    argument = UDFArgument(kind=UDFArgumentKind.TABLE, value=None)
    with pytest.raises(TypeError):
        argument.kind = UDFArgumentKind.LITERAL

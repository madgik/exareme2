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
    UDFArgument,
)
from mipengine.node_tasks_DTOs import TableType


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


def test_table_view_error():
    with pytest.raises(ValidationError):
        TableView(datasets=[34, "bar", "baz"], columns=[], filter=[])


def test_table_data_error():
    with pytest.raises(ValidationError):
        TableData(schema_="foo", data_=34)


def test_table_data():
    with pytest.raises(ValidationError):
        TableData(
            schema_="this is not a TableSchema object",
            data_="and this is not a list of tuples",
        )


def test_udf_argument():
    with pytest.raises(ValidationError):
        UDFArgument(kind="Not a UDFArgumentKind", value="this can be anything")

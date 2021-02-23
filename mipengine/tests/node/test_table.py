import pymonetdb
import pytest

from mipengine.node.node import app
from mipengine.node.tasks.data_classes import ColumnInfo
from mipengine.node.tasks.data_classes import TableData
from mipengine.node.tasks.data_classes import TableSchema

create_table = app.signature('mipengine.node.tasks.tables.create_table')
get_tables = app.signature('mipengine.node.tasks.tables.get_tables')
get_table_data = app.signature('mipengine.node.tasks.tables.get_table_data')
get_table_schema = app.signature('mipengine.node.tasks.tables.get_table_schema')
clean_up = app.signature('mipengine.node.tasks.common.clean_up')


def test_tables():
    context_id_1 = "regrEssion"
    schema = TableSchema([ColumnInfo("col1", "INT"), ColumnInfo("col2", "FLOAT"), ColumnInfo("col3", "TEXT")])
    json_schema = schema.to_json()
    table_1_name = create_table.delay(context_id_1, str(pymonetdb.uuid.uuid1()).replace("-", ""), json_schema).get()
    tables = get_tables.delay(context_id_1).get()

    assert table_1_name in tables
    table_data = get_table_data.delay(table_1_name).get()
    object_table_data = TableData.from_json(table_data)
    assert object_table_data.data == []
    assert object_table_data.schema == schema

    context_id_2 = "HISTOGRAMS"
    table_2_name = create_table.delay(context_id_2, str(pymonetdb.uuid.uuid1()).replace("-", ""), json_schema).get()
    tables = get_tables.delay(context_id_2).get()
    assert table_2_name in tables

    schema_result_json = get_table_schema.delay(table_2_name).get()
    object_schema_result = TableSchema.from_json(schema_result_json)
    assert object_schema_result == schema

    clean_up.delay(context_id_1.lower()).get()
    clean_up.delay(context_id_2.lower()).get()


def test_sql_injection_get_table_data():
    with pytest.raises(ValueError):
        get_table_data.delay("drop table data;").get()


def test_sql_injection_get_tables():
    with pytest.raises(ValueError):
        get_tables.delay("drop table data;").get()


def test_sql_injection_get_table_schema():
    with pytest.raises(ValueError):
        get_table_schema.delay("drop table data;").get()


def test_sql_injection_create_table_context_id():
    with pytest.raises(ValueError):
        context_id = "drop table data;"
        schema = TableSchema([ColumnInfo("col1", "INT"), ColumnInfo("col2", "FLOAT"), ColumnInfo("col3", "TEXT")])
        json_schema = schema.to_json()
        create_table.delay(context_id, str(pymonetdb.uuid.uuid1()).replace("-", ""), json_schema).get()


def test_sql_injection_create_table_uuid():
    with pytest.raises(ValueError):
        context_id = "HISTOGRAMS"
        schema = TableSchema([ColumnInfo("col1", "INT"), ColumnInfo("col2", "FLOAT"), ColumnInfo("col3", "TEXT")])
        json_schema = schema.to_json()
        create_table.delay(context_id, "drop table data;", json_schema).get()


def test_sql_injection_create_table_TableSchema_name():
    with pytest.raises(ValueError):
        context_id = "HISTOGRAMS"
        schema = TableSchema([ColumnInfo("drop table data;", "INT"), ColumnInfo("col2", "FLOAT"), ColumnInfo("col3", "TEXT")])
        json_schema = schema.to_json()
        create_table.delay(context_id, str(pymonetdb.uuid.uuid1()).replace("-", ""), json_schema).get()


def test_sql_injection_create_table_TableSchema_type():
    with pytest.raises(TypeError):
        context_id = "HISTOGRAMS"
        schema = TableSchema([ColumnInfo("col1", "drop table data;"), ColumnInfo("col2", "FLOAT"), ColumnInfo("col3", "TEXT")])
        json_schema = schema.to_json()
        create_table.delay(context_id, str(pymonetdb.uuid.uuid1()).replace("-", ""), json_schema).get()
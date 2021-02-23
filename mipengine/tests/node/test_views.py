import json

import pymonetdb
import pytest

from mipengine.node.node import app
from mipengine.node.tasks.data_classes import ColumnInfo
from mipengine.node.tasks.data_classes import TableData
from mipengine.node.tasks.data_classes import TableSchema

create_table = app.signature('mipengine.node.tasks.tables.create_table')
create_view = app.signature('mipengine.node.tasks.views.create_view')
get_views = app.signature('mipengine.node.tasks.views.get_views')
get_view_data = app.signature('mipengine.node.tasks.views.get_view_data')
get_view_schema = app.signature('mipengine.node.tasks.views.get_view_schema')
clean_up = app.signature('mipengine.node.tasks.common.clean_up')


def test_views():
    context_id = "regrEssion"
    columns = ["subjectcode", "dataset", "subjectvisitid", "subjectvisitdate"]
    datasets = ["edsd"]
    table_1_name = create_view.delay(context_id, str(pymonetdb.uuid.uuid1()).replace("-", ""), json.dumps(columns),
                                     json.dumps(datasets)).get()
    tables = get_views.delay(context_id).get()
    assert table_1_name in tables
    schema = TableSchema([
        ColumnInfo('subjectcode', 'clob'),
        ColumnInfo('dataset', 'clob'),
        ColumnInfo('subjectvisitid', 'clob'),
        ColumnInfo('subjectvisitdate', 'clob')])

    schema_result = get_view_schema.delay(table_1_name).get()
    object_schema_result = TableSchema.from_json(schema_result)
    assert object_schema_result == schema
    table_data_json = get_view_data.delay(table_1_name).get()
    table_data = TableData.from_json(table_data_json)
    assert table_data.data is not []
    assert table_data.schema == schema

    clean_up.delay(context_id.lower()).get()


def test_sql_injection_get_view_data():
    with pytest.raises(ValueError):
        get_view_data.delay("drop table data;").get()


def test_sql_injection_get_views():
    with pytest.raises(ValueError):
        get_views.delay("drop table data;").get()


def test_sql_injection_get_view_schema():
    with pytest.raises(ValueError):
        get_view_schema.delay("drop table data;").get()


def test_sql_injection_create_view_context_id():
    with pytest.raises(ValueError):
        context_id = "drop table data;"
        columns = ["subjectcode", "dataset", "subjectvisitid", "subjectvisitdate"]
        datasets = ["edsd"]
        create_view.delay(context_id, str(pymonetdb.uuid.uuid1()).replace("-", ""), json.dumps(columns),
                          json.dumps(datasets)).get()


def test_sql_injection_create_view_columns():
    with pytest.raises(ValueError):
        context_id = "regrEssion"
        columns = ["drop table data;", "dataset", "subjectvisitid", "subjectvisitdate"]
        datasets = ["edsd"]
        create_view.delay(context_id, str(pymonetdb.uuid.uuid1()).replace("-", ""), json.dumps(columns),
                          json.dumps(datasets)).get()


def test_sql_injection_create_view_datasets():
    with pytest.raises(ValueError):
        context_id = "regrEssion"
        columns = ["subjectcode", "dataset", "subjectvisitid", "subjectvisitdate"]
        datasets = ["drop table data;"]
        create_view.delay(context_id, str(pymonetdb.uuid.uuid1()).replace("-", ""), json.dumps(columns),
                          json.dumps(datasets)).get()


def test_sql_injection_create_view_uuid():
    with pytest.raises(ValueError):
        context_id = "regrEssion"
        columns = ["subjectcode", "dataset", "subjectvisitid", "subjectvisitdate"]
        datasets = ["edsd"]
        create_view.delay(context_id, "drop table data;", json.dumps(columns),
                          json.dumps(datasets)).get()

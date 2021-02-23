import json

import pymonetdb

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
    table_1_name = create_view.delay(context_id, str(pymonetdb.uuid.uuid1()).replace("-", ""), json.dumps(columns), json.dumps(datasets)).get()
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
    assert table_data.columns == schema

    clean_up.delay(context_id.lower()).get()

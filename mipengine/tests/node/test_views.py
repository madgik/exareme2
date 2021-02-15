import json
from mipengine.node.monetdb_interface.common import connection
from mipengine.node.monetdb_interface.common import cursor
from mipengine.node.node import app
from mipengine.node.tasks.data_classes import ColumnInfo
from mipengine.node.tasks.data_classes import TableData
import os

create_table = app.signature('mipengine.node.tasks.tables.create_table')
create_view = app.signature('mipengine.node.tasks.views.create_view')
get_views = app.signature('mipengine.node.tasks.views.get_views')
get_view_data = app.signature('mipengine.node.tasks.views.get_view_data')
get_view_schema = app.signature('mipengine.node.tasks.views.get_view_schema')
get_view_schema = app.signature('mipengine.node.tasks.views.get_view_schema')
clean_up = app.signature('mipengine.node.tasks.common.clean_up')


def test_views():
    setup_data_table()
    context_id = "regrEssion"
    columns = ["subjectcode", "dataset", "subjectvisitid", "subjectvisitdate"]
    datasets = ["edsd"]
    table_1_name = create_view.delay(context_id, json.dumps(columns), json.dumps(datasets)).get()
    tables = get_views.delay(context_id).get()
    assert table_1_name in tables
    schema = [
        ColumnInfo('subjectcode', 'clob'),
        ColumnInfo('dataset', 'clob'),
        ColumnInfo('subjectvisitid', 'clob'),
        ColumnInfo('subjectvisitdate', 'clob')]
    schema_result = get_view_schema.delay(table_1_name).get()
    object_schema_result = ColumnInfo.schema().loads(schema_result, many=True)
    assert object_schema_result == schema
    table_data_json = get_view_data.delay(table_1_name).get()
    table_data = TableData.from_json(table_data_json)
    assert table_data.data != []
    assert table_data.schema == schema

    clean_up.delay(context_id).get()
    delete_data_table()


def delete_data_table():
    cursor.execute("DROP TABLE data cascade;")
    connection.commit()


def setup_data_table():
    script_dir = os.path.dirname(__file__)
    data_table_creation_path = os.path.join(script_dir, "data_table_creation.txt")
    columns = open(data_table_creation_path, "r")
    cursor.execute(columns.read())
    data_path = os.path.join(script_dir, "data.txt")
    data = open(data_path, "r")
    cursor.execute(data.read())
    connection.commit()

import json
import unittest


from mipengine.node.monetdb_interface.common import connection
from mipengine.node.monetdb_interface.common import cursor
from mipengine.node.node import app
from mipengine.node.tasks.data_classes import ColumnInfo
from mipengine.node.tasks.data_classes import TableData

create_table = app.signature('mipengine.node.tasks.tables.create_table')
create_view = app.signature('mipengine.node.tasks.views.create_view')
get_views = app.signature('mipengine.node.tasks.views.get_views')
get_view_data = app.signature('mipengine.node.tasks.views.get_view_data')
get_view_schema = app.signature('mipengine.node.tasks.views.get_view_schema')
get_view_schema = app.signature('mipengine.node.tasks.views.get_view_schema')
clean_up = app.signature('mipengine.node.tasks.views.clean_up')


def test_views():
    # setup_data_table()
    context_id_1 = "regrEssion"
    columns = ["subjectcode", "dataset", "subjectvisitid", "subjectvisitdate"]
    datasets = ["edsd"]
    table_1_name = create_view.delay(context_id_1, json.dumps(columns), json.dumps(datasets)).get()
    tables = get_views.delay(context_id_1).get()
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

    assert clean_up.delay(context_id_1).get() == 0
    assert clean_up.delay().get() == 0


def setup_data_table():
    columns = open("/home/kostas/Desktop/MIP-Engine/mipengine/tests/node/columns.txt", "r")
    cursor.execute(columns.read())
    data = open("/home/kostas/Desktop/MIP-Engine/mipengine/tests/node/data.txt", "r")
    cursor.execute(data.read())
    connection.commit()


if __name__ == "__main__":
    unittest.main()

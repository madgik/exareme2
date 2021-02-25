import pymonetdb
import pytest

from mipengine.node.tasks.data_classes import ColumnInfo
from mipengine.node.tasks.data_classes import TableInfo
from mipengine.node.tasks.data_classes import TableSchema
from mipengine.tests.node.set_up_nodes import celery_local_node_1
from mipengine.tests.node.set_up_nodes import celery_local_node_2

create_table = celery_local_node_1.signature('mipengine.node.tasks.tables.create_table')
clean_up_node1 = celery_local_node_1.signature('mipengine.node.tasks.common.clean_up')
create_remote_table = celery_local_node_2.signature('mipengine.node.tasks.remote_tables.create_remote_table')
get_remote_tables = celery_local_node_2.signature('mipengine.node.tasks.remote_tables.get_remote_tables')
clean_up_node2 = celery_local_node_2.signature('mipengine.node.tasks.common.clean_up')

context_id = "regrEssion"
url = 'mapi:monetdb://192.168.1.147:50000/db'
schema = TableSchema([ColumnInfo("col1", "INT"), ColumnInfo("col2", "FLOAT"), ColumnInfo("col3", "TEXT")])
json_schema = schema.to_json()
test_table_name = create_table.delay(context_id, str(pymonetdb.uuid.uuid1()).replace("-", ""), json_schema).get()
table_info = TableInfo(test_table_name, schema)
table_info_json = table_info.to_json()


def test_remote_tables():
    create_remote_table.delay(table_info_json, url).get()
    tables = get_remote_tables.delay(context_id).get()
    assert test_table_name in tables

    clean_up_node1.delay(context_id.lower()).get()
    clean_up_node2.delay(context_id.lower()).get()


def test_sql_injection_get_remote_tables():
    with pytest.raises(ValueError):
        get_remote_tables.delay("drop table data;").get()


def test_sql_injection_create_remote_table_table_schema_name():
    with pytest.raises(ValueError):
        invalid_schema = TableSchema(
            [ColumnInfo("drop table data;", "INT"), ColumnInfo("col2", "FLOAT"), ColumnInfo("col3", "TEXT")])
        invalid_table_info = TableInfo(test_table_name, invalid_schema)
        invalid_table_info_json = invalid_table_info.to_json()
        create_remote_table.delay(invalid_table_info_json, url).get()


def test_sql_injection_table_schema_type():
    with pytest.raises(TypeError):
        invalid_schema = TableSchema(
            [ColumnInfo("col1", "drop table data;"), ColumnInfo("col2", "FLOAT"), ColumnInfo("col3", "TEXT")])
        invalid_table_info = TableInfo(test_table_name, invalid_schema)
        invalid_table_info_json = invalid_table_info.to_json()
        create_remote_table.delay(invalid_table_info_json, url).get()


def test_sql_injection_create_remote_table_url():
    with pytest.raises(ValueError):
        create_remote_table.delay(table_info_json, "drop table data;").get()

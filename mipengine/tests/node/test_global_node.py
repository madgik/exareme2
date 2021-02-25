import pymonetdb

from mipengine.node.tasks.data_classes import ColumnInfo
from mipengine.node.tasks.data_classes import TableInfo
from mipengine.node.tasks.data_classes import TableSchema
from mipengine.tests.node.node_db_connections import get_node_db_connection
from mipengine.tests.node.set_up_nodes import celery_global_node
from mipengine.tests.node.set_up_nodes import celery_local_node_1
from mipengine.tests.node.set_up_nodes import celery_local_node_2

create_table_local_node_1 = celery_local_node_1.signature('mipengine.node.tasks.tables.create_table')
create_table_local_node_2 = celery_local_node_2.signature('mipengine.node.tasks.tables.create_table')
clean_up_node1 = celery_local_node_1.signature('mipengine.node.tasks.common.clean_up')
clean_up_node2 = celery_local_node_2.signature('mipengine.node.tasks.common.clean_up')
create_remote_table = celery_global_node.signature('mipengine.node.tasks.remote_tables.create_remote_table')
get_remote_tables = celery_global_node.signature('mipengine.node.tasks.remote_tables.get_remote_tables')
create_merge_table = celery_global_node.signature('mipengine.node.tasks.merge_tables.create_merge_table')
get_merge_tables = celery_global_node.signature('mipengine.node.tasks.merge_tables.get_merge_tables')
clean_up_global = celery_global_node.signature('mipengine.node.tasks.common.clean_up')


def insert_data_into_local_node_db(node_id: str, table_name: str):
    connection = get_node_db_connection(node_id)
    cursor = connection.cursor()

    cursor.execute(f"INSERT INTO {table_name} VALUES (1, 1.2, {table_name})")
    connection.commit()
    connection.close()


def test_global_node():
    context_id = "regrEssion"
    clean_up_global.delay(context_id).get()
    clean_up_node1.delay(context_id).get()
    clean_up_node2.delay(context_id).get()
    url_local_node_1 = 'mapi:monetdb://192.168.1.147:50001/db'
    url_local_node_2 = 'mapi:monetdb://192.168.1.147:50002/db'
    schema = TableSchema([ColumnInfo("col1", "INT"), ColumnInfo("col2", "FLOAT"), ColumnInfo("col3", "TEXT")])
    json_schema = schema.to_json()

    # Setup local node 1
    local_node_1_table_name = create_table_local_node_1.delay(context_id, str(pymonetdb.uuid.uuid1()).replace("-", ""),
                                                              json_schema).get()
    # insert_data_into_local_node_db('local_node_1', local_node_1_table_name)
    table_info_local_1 = TableInfo(local_node_1_table_name, schema)
    table_info_json_1 = table_info_local_1.to_json()

    # Setup local node 2
    local_node_2_table_name = create_table_local_node_2.delay(context_id, str(pymonetdb.uuid.uuid1()).replace("-", ""),
                                                              json_schema).get()
    # insert_data_into_local_node_db('local_node_2', local_node_2_table_name)
    table_info_local_2 = TableInfo(local_node_2_table_name, schema)
    table_info_json_2 = table_info_local_2.to_json()

    # Setup remote tables

    create_remote_table.delay(table_info_json_1, url_local_node_1).get()
    create_remote_table.delay(table_info_json_2, url_local_node_2).get()
    remote_tables = get_remote_tables.delay(context_id).get()
    assert local_node_1_table_name in remote_tables
    assert local_node_2_table_name in remote_tables

    # Setup merge table
    merge_table_name = create_merge_table.delay(context_id, str(pymonetdb.uuid.uuid1()).replace("-", ""),
                                                remote_tables).get()
    merge_tables = get_merge_tables.delay(context_id).get()
    assert merge_table_name in merge_tables

    clean_up_global.delay(context_id.lower()).get()
    clean_up_node1.delay(context_id.lower()).get()
    clean_up_node2.delay(context_id.lower()).get()


test_global_node()

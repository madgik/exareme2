from celery import Celery

from mipengine.node.config.config_parser import Config
from mipengine.node.tasks.data_classes import ColumnInfo
from mipengine.node.tasks.data_classes import TableInfo

config = Config().config
user = config["rabbitmq"]["user"]
password = config["rabbitmq"]["password"]
vhost = config["rabbitmq"]["vhost"]
node1 = Celery('mipengine.node',
               broker=f'amqp://{user}:{password}@{"127.0.0.1"}:{5672}/{vhost}',
               backend='rpc://',
               include=['mipengine.node.tasks.tables', 'mipengine.node.tasks.remote_tables',
                        'mipengine.node.tasks.merge_tables', 'mipengine.node.tasks.common'])

node2 = Celery('mipengine.node',
               broker=f'amqp://{user}:{password}@{"127.0.0.1"}:{5673}/{vhost}',
               backend='rpc://',
               include=['mipengine.node.tasks.tables', 'mipengine.node.tasks.remote_tables',
                        'mipengine.node.tasks.merge_tables', 'mipengine.node.tasks.common'])

create_table = node1.signature('mipengine.node.tasks.tables.create_table')
create_remote_table = node2.signature('mipengine.node.tasks.remote_tables.create_remote_table')
get_remote_tables = node2.signature('mipengine.node.tasks.remote_tables.get_remote_tables')
clean_up_node1 = node1.signature('mipengine.node.tasks.common.clean_up')
clean_up_node2 = node2.signature('mipengine.node.tasks.common.clean_up')


def test_remote_tables():
    context_id = "regrEssion"
    schema = [ColumnInfo("col1", "INT"), ColumnInfo("col2", "FLOAT"), ColumnInfo("col3", "TEXT")]
    json_schema = ColumnInfo.schema().dumps(schema, many=True)
    test_table_name = create_table.delay(context_id, json_schema).get()
    url = 'mapi:monetdb://192.168.1.147:50000/db'
    table_info = TableInfo(test_table_name, schema)
    table_info_json = table_info.to_json()
    create_remote_table.delay(table_info_json, url).get()
    tables = get_remote_tables.delay(context_id).get()
    assert test_table_name in tables

    clean_up_node1.delay(context_id).get()
    clean_up_node2.delay(context_id).get()
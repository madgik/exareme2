import unittest

from celery import Celery

from mipengine.node.config.config_parser import Config
from mipengine.node.tasks.data_classes import ColumnInfo
from mipengine.node.tasks.data_classes import TableInfo

config = Config().config
ip = config["rabbitmq"]["ip"]
port = config["rabbitmq"]["port"]
user = config["rabbitmq"]["user"]
password = config["rabbitmq"]["password"]
vhost = config["rabbitmq"]["vhost"]
node1 = Celery('mipengine.node',
               broker=f'amqp://{user}:{password}@{ip}:{5672}/{vhost}',
               backend='rpc://',
               include=['mipengine.node.tasks.tables', 'mipengine.node.tasks.remote_tables',
                        'mipengine.node.tasks.merge_tables'])

node2 = Celery('mipengine.node',
               broker=f'amqp://{user}:{password}@{ip}:{5673}/{vhost}',
               backend='rpc://',
               include=['mipengine.node.tasks.tables', 'mipengine.node.tasks.remote_tables',
                        'mipengine.node.tasks.merge_tables'])

create_table = node1.signature('mipengine.node.tasks.tables.create_table')
clean_up_node1 = node1.signature('mipengine.node.tasks.tables.clean_up')
create_remote_table = node2.signature('mipengine.node.tasks.remote_tables.create_remote_table')
clean_up_node2 = node2.signature('mipengine.node.tasks.remote_tables.clean_up')
get_remote_tables = node2.signature('mipengine.node.tasks.remote_tables.get_remote_tables')


def test_remote_tables():
    context_id = "regrEssion"
    schema = [ColumnInfo("col1", "INT"), ColumnInfo("col2", "FLOAT"), ColumnInfo("col3", "TEXT")]
    json_schema = ColumnInfo.schema().dumps(schema, many=True)
    test_table_name = create_table.delay(context_id, json_schema).get()
    url = 'mapi:monetdb://192.168.1.147:50000/db'
    table_info = TableInfo(test_table_name, schema)
    table_info_json = table_info.to_json()
    assert create_remote_table.delay(table_info_json, url).get() == 0
    tables = get_remote_tables.delay(context_id).get()
    assert test_table_name in tables

    assert clean_up_node1.delay(context_id).get() == 0
    assert clean_up_node2.delay(context_id).get() == 0


if __name__ == "__main__":
    unittest.main()

from pprint import pprint

from celery import Celery

from mipengine.config import Config
from mipengine.worker.tasks.data_classes import TableInfo

config = Config().config
ip = config["rabbitmq"]["ip"]
port = config["rabbitmq"]["port"]
user = config["rabbitmq"]["user"]
password = config["rabbitmq"]["password"]
vhost = config["rabbitmq"]["vhost"]
node1 = Celery('worker',
               broker=f'amqp://{user}:{password}@{ip}:{5672}/{vhost}',
               backend='rpc://',
               include=['worker.tasks.tables', 'worker.tasks.remote_tables', 'worker.tasks.merge_tables'])

node2 = Celery('worker',
               broker=f'amqp://{user}:{password}@{ip}:{5673}/{vhost}',
               backend='rpc://',
               include=['worker.tasks.tables', 'worker.tasks.remote_tables', 'worker.tasks.merge_tables'])


create_table = node1.signature('worker.tasks.tables.create_table')
delete_table = node1.signature('worker.tasks.tables.delete_table')
table_1 = create_table.delay(
    [{"name": "col1", "type": "INT"}, {"name": "col2", "type": "FLOAT"},
     {"name": "col3", "type": "TEXT"}], "test_for_remote_table").get()
pprint(f"Created table. Response: \n{table_1}")

test_table_name = TableInfo.from_json(table_1).name

print('Running worker_tests for remote tables...')

create_remote_table = node2.signature('worker.tasks.remote_tables.create_remote_table')
delete_remote_table = node2.signature('worker.tasks.remote_tables.delete_remote_table')
get_remote_table_data = node2.signature('worker.tasks.remote_tables.get_remote_table_data')
get_remote_tables_info = node2.signature('worker.tasks.remote_tables.get_remote_tables_info')


# Creating a fake table and retrieve it's data and schema afterwards.
url = 'mapi:monetdb://192.168.1.147:50000/db'
remote_table = create_remote_table.delay(
    [{"name": "col1", "type": "INT"}, {"name": "col2", "type": "FLOAT"},
     {"name": "col3", "type": "TEXT"}], test_table_name, url).get()
pprint(f"Created remote table. Response: \n{remote_table}")

regression_table = TableInfo.from_json(remote_table)
print(regression_table)
table_data = get_remote_table_data.delay(regression_table.name).get()
pprint(f"Got remote table data. Response: \n{table_data}")

tables = get_remote_tables_info.delay().get()
pprint(f"Got tables. Response: \n{tables}")

remote_table_name = TableInfo.from_json(remote_table).name
delete_remote_table_result = delete_remote_table.delay(remote_table_name).get()
pprint(f'Deleted remote table. Response: \n {delete_remote_table_result}')

delete_table_result = delete_table.delay(remote_table_name).get()
pprint(f'Deleted table. Response: \n {delete_table_result}')

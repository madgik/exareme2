from pprint import pprint

from celery import Celery

# TODO Convert to an actual test framework
from worker.config.config_parser import Config
from worker.tasks.data_classes import TableInfo

config = Config().config
ip = config["rabbitmq"]["ip"]
port = config["rabbitmq"]["port"]
user = config["rabbitmq"]["user"]
password = config["rabbitmq"]["password"]
vhost = config["rabbitmq"]["vhost"]
node1 = Celery('worker',
               broker=f'amqp://{user}:{password}@{ip}:{port}/{vhost}',
               backend='rpc://',
               include=['worker.tasks.tables'])

create_table = node1.signature('worker.tasks.tables.create_table')
delete_table = node1.signature('worker.tasks.tables.delete_table')
get_table_data = node1.signature('worker.tasks.tables.get_table_data')
get_tables_info = node1.signature('worker.tasks.tables.get_tables_info')

print('Running tests...')

# Creating a fake table and retrieve it's data and schema afterwards.
table_1 = create_table.delay(
    [{"name": "col1", "type": "INT"}, {"name": "col2", "type": "FLOAT"},
     {"name": "col3", "type": "TEXT"}], "regression_211232433443").get()
pprint(f"Created table1. Response: \n{table_1}")

regression_table = TableInfo.from_json(table_1)
table_data = get_table_data.delay(regression_table.name).get()
pprint(f"Got table data. Response: \n{table_data}")

# Creating a fake table and retrieve it's data and schema afterwards.
table_2 = create_table.delay(
    [{"name": "col1", "type": "INT"}, {"name": "col2", "type": "FLOAT"},
     {"name": "col3", "type": "TEXT"}], "histograms_211232433443").get()
pprint(f"Created table2. Response: \n{table_2}")

tables = get_tables_info.delay().get()
pprint(f"Got tables. Response: \n{tables}")

table_1_name = TableInfo.from_json(table_1).name
table_2_name = TableInfo.from_json(table_2).name
result_1 = delete_table.delay(table_1_name).get()
pprint(f'Deleted table_1. Response: \n {result_1}')
result_2 = delete_table.delay(table_2_name).get()
pprint(f'Deleted table_2. Response: \n {result_2}')

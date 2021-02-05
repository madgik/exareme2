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
               broker=f'amqp://{user}:{password}@{ip}:{port}/{vhost}',
               backend='rpc://',
               include=['worker.tasks.merge_tables'])

create_merge_table = node1.signature('worker.tasks.merge_tables.create_merge_table')
delete_merge_table = node1.signature('worker.tasks.merge_tables.delete_merge_table')
get_merge_table_data = node1.signature('worker.tasks.merge_tables.get_merge_table_data')
get_merge_tables_info = node1.signature('worker.tasks.merge_tables.get_merge_tables_info')

print('Running tests for merge tables...')

# Creating a fake table and retrieve it's data and schema afterwards.
merge_table_1 = create_merge_table.delay(
    [{"name": "col1", "type": "INT"}, {"name": "col2", "type": "FLOAT"},
     {"name": "col3", "type": "TEXT"}], "regression_211232433443",
    ["table_801dab92654011eb8d1c830ae9bb4267_regression_211232433443",
     "table_801dab93654011eb8d1c830ae9bb4267_histograms_211232433443"]).get()
pprint(f"Created merge table1. Response: \n{merge_table_1}")

regression_table = TableInfo.from_json(merge_table_1)
table_data = get_merge_table_data.delay(regression_table.name).get()
pprint(f"Got table data. Response: \n{table_data}")

# Creating a fake table and retrieve it's data and schema afterwards.
merge_table_2 = create_merge_table.delay(
    [{"name": "col1", "type": "INT"}, {"name": "col2", "type": "FLOAT"},
     {"name": "col3", "type": "TEXT"}], "histograms_211232433443",
    ["table_9273f292654011eb8d1c830ae9bb4267_regression_211232433443",
     "table_9273f293654011eb8d1c830ae9bb4267_histograms_211232433443"]).get()
pprint(f"Created table2. Response: \n{merge_table_2}")

tables = get_merge_tables_info.delay().get()
pprint(f"Got tables. Response: \n{tables}")

table_1_name = TableInfo.from_json(merge_table_1).name
table_2_name = TableInfo.from_json(merge_table_2).name
result_1 = delete_merge_table.delay(table_1_name).get()
pprint(f'Deleted table_1. Response: \n {result_1}')
result_2 = delete_merge_table.delay(table_2_name).get()
pprint(f'Deleted table_2. Response: \n {result_2}')

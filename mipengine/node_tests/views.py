from pprint import pprint

from celery import Celery

# TODO Convert to an actual test framework
from mipengine.config.config_parser import Config
from mipengine.node.tasks.data_classes import TableInfo

config = Config().config
ip = config["rabbitmq"]["ip"]
port = config["rabbitmq"]["port"]
user = config["rabbitmq"]["user"]
password = config["rabbitmq"]["password"]
vhost = config["rabbitmq"]["vhost"]
node1 = Celery('mipengine.node',
               broker=f'amqp://{user}:{password}@{ip}:{port}/{vhost}',
               backend='rpc://',
               include=['mipengine.node.tasks.views'])

create_view = node1.signature('mipengine.node.tasks.views.create_view')
clean_up = node1.signature('mipengine.node.tasks.views.clean_up')
get_view_schema = node1.signature('mipengine.node.tasks.views.get_view_schema')
get_view_data = node1.signature('mipengine.node.tasks.views.get_view_data')
get_views = node1.signature('mipengine.node.tasks.views.get_views')

print('Running node_tests...')

# Creating a fake view and retrieve it's data and schema afterwards.
context_id_1 = "regrEssion"
table_1_name = create_view.delay(context_id_1, ["name"], ["ppmi"]).get()
pprint(f"Created table1. Response: \n{table_1_name}")

table_data = get_view_data.delay(table_1_name).get()
pprint(f"Got table data. Response: \n{table_data}")

# Creating a fake table and retrieve it's data and schema afterwards.
context_id_2 = "HISTOGRAMS"
table_2_name = create_view.delay(context_id_2, ["name", "pathology"], ["ppmi", "mentalhealth"]).get()
pprint(f"Created table2. Response: \n{table_2_name}")

schema = get_view_schema.delay(table_2_name).get()
pprint(f"Got schema. Response: \n{schema}")

tables = get_views.delay(context_id_1).get()
pprint(f"Got tables with context: {context_id_1}. Response: \n{tables}")
tables = get_views.delay(context_id_2).get()
pprint(f"Got tables with context: {context_id_2}. Response: \n{tables}")

print('Optional empty database its important for our sanity...')
clean_up.delay(context_id_1)
clean_up.delay(context_id_2)

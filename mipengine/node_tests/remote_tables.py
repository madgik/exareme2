from pprint import pprint

from celery import Celery

from mipengine.config.config_parser import Config
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
clean_up_node1 = node1.signature('mipengine.node.tasks.remote_tables.clean_up')

context_id = "regression"
test_table_name = create_table.delay(context_id,
                                     [{"name": "col1", "type": "INT"}, {"name": "col2", "type": "FLOAT"},
                                      {"name": "col3", "type": "TEXT"}]).get()
pprint(f"Created table. Response: \n{test_table_name}")

print('Running node_tests for remote tables...')

create_remote_table = node2.signature('mipengine.node.tasks.remote_tables.create_remote_table')
clean_up_node2 = node2.signature('mipengine.node.tasks.remote_tables.clean_up')
get_remote_tables = node2.signature('mipengine.node.tasks.remote_tables.get_remote_tables')

# Creating a fake table and retrieve it's data and schema afterwards.
url = 'mapi:monetdb://192.168.1.147:50000/db'
create_remote_table.delay(test_table_name,
                          [{"name": "col1", "type": "INT"}, {"name": "col2", "type": "FLOAT"},
                           {"name": "col3", "type": "TEXT"}], url).get()

tables = get_remote_tables.delay(context_id).get()
pprint(f"Got tables. Response: \n{tables}")

print('Optional empty database its important for our sanity...')
clean_up_node1.delay(context_id)
clean_up_node2.delay(context_id)

from celery import Celery

from mipengine.common.node_catalog import node_catalog
from mipengine.node.config.config_parser import config

global_node = node_catalog.get_global_node()
local_node_1 = node_catalog.get_local_node_data("local_node_1")
local_node_2 = node_catalog.get_local_node_data("local_node_2")

user = config["rabbitmq"]["user"]
password = config["rabbitmq"]["password"]
vhost = config["rabbitmq"]["vhost"]
celery_global_node = Celery('mipengine.node',
                            broker=f'amqp://{user}:{password}@{global_node.rabbitmqURL}/{vhost}',
                            backend='rpc://',
                            include=['mipengine.node.tasks.tables', 'mipengine.node.tasks.remote_tables',
                                     'mipengine.node.tasks.merge_tables', 'mipengine.node.tasks.common'])

celery_local_node_1 = Celery('mipengine.node',
                             broker=f'amqp://{user}:{password}@{local_node_1.rabbitmqURL}/{vhost}',
                             backend='rpc://',
                             include=['mipengine.node.tasks.tables', 'mipengine.node.tasks.remote_tables',
                                      'mipengine.node.tasks.merge_tables', 'mipengine.node.tasks.common'])

celery_local_node_2 = Celery('mipengine.node',
                             broker=f'amqp://{user}:{password}@{local_node_2.rabbitmqURL}/{vhost}',
                             backend='rpc://',
                             include=['mipengine.node.tasks.tables', 'mipengine.node.tasks.remote_tables',
                                      'mipengine.node.tasks.merge_tables', 'mipengine.node.tasks.common'])

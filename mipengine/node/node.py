from celery import Celery

from mipengine.common.node_catalog import node_catalog
from mipengine.node.config.config_parser import config


global_node = node_catalog.get_global_node()
if global_node.nodeId == config.get("node", "identifier"):
    node = global_node
else:
    node = node_catalog.get_local_node_data(config.get("node", "identifier"))

rabbitmqURL = node.rabbitmqURL
user = config.get("rabbitmq", "user")
password = config.get("rabbitmq", "password")
vhost = config.get("rabbitmq", "vhost")

app = Celery('mipengine.node',
             broker=f'amqp://{user}:{password}@{rabbitmqURL}/{vhost}',
             backend='rpc://',
             include=['mipengine.node.tasks.tables', 'mipengine.node.tasks.remote_tables',
                      'mipengine.node.tasks.merge_tables', 'mipengine.node.tasks.views', 'mipengine.node.tasks.common'])

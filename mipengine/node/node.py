from celery import Celery

from mipengine.common.node_catalog import node_catalog
from mipengine.node import config as node_config

# Send information to node_catalog
node_catalog.set_node(
    node_id=node_config.identifier,
    monetdb_ip=node_config.monetdb.ip,
    monetdb_port=node_config.monetdb.port,
    rabbitmq_ip=node_config.rabbitmq.ip,
    rabbitmq_port=node_config.rabbitmq.port,
)

rabbitmq_credentials = node_config.rabbitmq.user + ":" + node_config.rabbitmq.password
rabbitmq_socket_addr = node_config.rabbitmq.ip + ":" + str(node_config.rabbitmq.port)
vhost = node_config.rabbitmq.vhost

celery = Celery(
    "mipengine.node",
    broker=f"amqp://{rabbitmq_credentials}@{rabbitmq_socket_addr}/{vhost}",
    backend="rpc://",
    include=[
        "mipengine.node.tasks.tables",
        "mipengine.node.tasks.remote_tables",
        "mipengine.node.tasks.merge_tables",
        "mipengine.node.tasks.views",
        "mipengine.node.tasks.common",
        "mipengine.node.tasks.udfs",
    ],
)

celery.conf.worker_concurrency = node_config.celery.worker_concurrency
celery.conf.task_soft_time_limit = node_config.celery.task_soft_time_limit
celery.conf.task_time_limit = node_config.celery.task_time_limit

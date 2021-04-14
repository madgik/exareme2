import argparse

from celery import Celery

from mipengine import config
from mipengine.common.node_catalog import node_catalog
from mipengine.node import celery_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-id", help="Current node identifier.", required=True)
    extra_args, celery_args = parser.parse_known_args()
    config.node.identifier = extra_args.node_id

global_node = node_catalog.get_global_node()
if global_node.nodeId == config.node.identifier:
    current_node = global_node
else:
    current_node = node_catalog.get_local_node(config.node.identifier)

rabbitmqURL = current_node.rabbitmqURL
user = config.rabbitmq.user
password = config.rabbitmq.password
vhost = config.rabbitmq.vhost

app = Celery(
    "mipengine.node",
    broker=f"amqp://{user}:{password}@{rabbitmqURL}/{vhost}",
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

app.config_from_object(celery_config)

if __name__ == "__main__":
    app.worker_main(celery_args)

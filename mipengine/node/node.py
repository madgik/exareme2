import logging

from celery import Celery
from celery import signals

from mipengine.node import config as node_config

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


@signals.setup_logging.connect
def setup_celery_logging(*args, **kwargs):
    logger = logging.getLogger()
    formatter = logging.Formatter(
        f"%(asctime)s - %(levelname)s - NODE - {node_config.role} - {node_config.identifier} - CELERY - FRAMEWORK - %(message)s"
    )

    # StreamHandler
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    logger.setLevel(node_config.log_level)


celery.conf.worker_concurrency = node_config.celery.worker_concurrency
celery.conf.task_soft_time_limit = node_config.celery.task_soft_time_limit
celery.conf.task_time_limit = node_config.celery.task_time_limit

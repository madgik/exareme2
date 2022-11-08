import logging

from celery import Celery
from celery import signals

from mipengine.celery_app_conf import configure_celery_app_to_use_priority_queue
from mipengine.node import config as node_config
from mipengine.node.node_logger import init_logger


class InvalidMinimumRowCountError(Exception):
    def __init__(self, node_identifier: str, minimum_row_count_val: int):
        self.message = f"minimum_row_count for node:{node_identifier} is set to the invalid value:{minimum_row_count_val}. Minimum row count must be a greater than zero integer."
        super().__init__(self.message)


minimum_row_count = node_config.privacy.minimum_row_count
if minimum_row_count < 1 or int(minimum_row_count) != minimum_row_count:
    raise InvalidMinimumRowCountError(
        node_identifier=node_config.identifier,
        minimum_row_count_val=minimum_row_count,
    )


rabbitmq_credentials = node_config.rabbitmq.user + ":" + node_config.rabbitmq.password
rabbitmq_socket_addr = node_config.rabbitmq.ip + ":" + str(node_config.rabbitmq.port)
vhost = node_config.rabbitmq.vhost

node_logger = init_logger("NODE INITIALIZATION")

node_logger.info("Creating the celery app...")
celery = Celery(
    "mipengine.node",
    broker=f"pyamqp://{rabbitmq_credentials}@{rabbitmq_socket_addr}/{vhost}",
    backend="rpc://",
    include=[
        "mipengine.node.tasks.tables",
        "mipengine.node.tasks.remote_tables",
        "mipengine.node.tasks.merge_tables",
        "mipengine.node.tasks.views",
        "mipengine.node.tasks.common",
        "mipengine.node.tasks.udfs",
        "mipengine.node.tasks.smpc",
    ],
)
node_logger.info("Celery app created.")


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

    logger.setLevel(node_config.framework_log_level)


celery.conf.worker_concurrency = node_config.celery.worker_concurrency

configure_celery_app_to_use_priority_queue(celery)

# TODO https://team-1617704806227.atlassian.net/browse/MIP-473
# celery.conf.task_soft_time_limit = node_config.celery.task_soft_time_limit
# celery.conf.task_time_limit = node_config.celery.task_time_limit

"""
After the node.py is imported the celery process is launched
and the connection with the broker (rabbitmq) is established.
If the connection cannot be established, no message is shown,
so we need to know it failed at this point.
"""
node_logger.info("Connecting with broker...")

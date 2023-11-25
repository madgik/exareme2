import logging

from celery import Celery
from celery import signals

from exareme2.celery_app_conf import configure_celery_app_to_use_priority_queue
from exareme2.node import config as node_config
from exareme2.node.logger import init_logger

rabbitmq_credentials = node_config.rabbitmq.user + ":" + node_config.rabbitmq.password
rabbitmq_socket_addr = node_config.rabbitmq.ip + ":" + str(node_config.rabbitmq.port)
vhost = node_config.rabbitmq.vhost

node_logger = init_logger("NODE INITIALIZATION")

node_logger.info("Creating the celery app...")
app = Celery(
    "exareme2.node.celery",
    broker=f"pyamqp://{rabbitmq_credentials}@{rabbitmq_socket_addr}/{vhost}",
    backend="rpc://",
    include=[
        "exareme2.node.celery_tasks.node_info",
        "exareme2.node.celery_tasks.views",
        "exareme2.node.celery_tasks.tables",
        "exareme2.node.celery_tasks.udfs",
        "exareme2.node.celery_tasks.smpc",
        "exareme2.node.celery_tasks.cleanup",
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


app.conf.worker_concurrency = node_config.celery.worker_concurrency

configure_celery_app_to_use_priority_queue(app)

"""
After the app.py is imported the celery process is launched
and the connection with the broker (rabbitmq) is established.
If the connection cannot be established, no message is shown,
so we need to know it failed at this point.
"""
node_logger.info("Connecting with broker...")

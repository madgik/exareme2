import logging
import queue

from celery import Celery
from celery import signals

from exareme2.celery_app_conf import configure_celery_app_to_use_priority_queue
from exareme2.worker import config as worker_config
from exareme2.worker.utils.logger import init_logger

# ----------------------- Flower1.19 workaround -----------------------
# Flower 1.19 imports `queue.Queue` and then annotates it as Queue[Optional[str]],
# which crashes on Python ≥3.9 because queue.Queue is not a generic type.
if not hasattr(queue.Queue, "__class_getitem__"):
    # make Queue[...] legal by returning the plain class
    queue.Queue.__class_getitem__ = classmethod(lambda cls, _type: cls)
# ----------------------------------------------------------------------

rabbitmq_credentials = (
    worker_config.rabbitmq.user + ":" + worker_config.rabbitmq.password
)
rabbitmq_socket_addr = (
    worker_config.rabbitmq.ip + ":" + str(worker_config.rabbitmq.port)
)
vhost = worker_config.rabbitmq.vhost

worker_logger = init_logger("WORKER INITIALIZATION")

worker_logger.info("Creating the celery app...")
app = Celery(
    "exareme2.worker",
    broker=f"pyamqp://{rabbitmq_credentials}@{rabbitmq_socket_addr}/{vhost}",
    backend="rpc://",
    include=[
        "exareme2.worker.worker_info.worker_info_api",
        "exareme2.worker.exareme2.views.views_api",
        "exareme2.worker.exareme2.tables.tables_api",
        "exareme2.worker.exareme2.udfs.udfs_api",
        "exareme2.worker.exareme2.smpc.smpc_api",
        "exareme2.worker.exareme2.cleanup.cleanup_api",
        "exareme2.worker.flower.starter.starter_api",
        "exareme2.worker.flower.cleanup.cleanup_api",
        "exareme2.worker.exaflow.udf.udf_api",
    ],
)
worker_logger.info("Celery app created.")


@signals.setup_logging.connect
def setup_celery_logging(*args, **kwargs):
    logger = logging.getLogger()
    formatter = logging.Formatter(
        f"%(asctime)s - %(levelname)s - CELERY FRAMEWORK - [{worker_config.federation}] - [exareme2-{worker_config.role.lower()}] - [{worker_config.identifier}] - [FRAMEWORK] - %(message)s"
    )

    # StreamHandler
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    logger.setLevel(worker_config.framework_log_level)


app.conf.worker_concurrency = worker_config.celery.worker_concurrency

configure_celery_app_to_use_priority_queue(app)

"""
After the app.py is imported the celery process is launched
and the connection with the broker (rabbitmq) is established.
If the connection cannot be established, no message is shown,
so we need to know it failed at this point.
"""
worker_logger.info("Connecting with broker...")

import logging

from celery import Celery
from celery import signals
from pythonjsonlogger import jsonlogger

from exareme2.celery_app_conf import configure_celery_app_to_use_priority_queue
from exareme2.worker import config as worker_config
from exareme2.worker.utils.logger import init_logger

rabbitmq_credentials = (
    f"{worker_config.rabbitmq.user}:{worker_config.rabbitmq.password}"
)
rabbitmq_socket_addr = f"{worker_config.rabbitmq.ip}:{worker_config.rabbitmq.port}"
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
    ],
)
worker_logger.info("Celery app created.")


# Custom JSON Formatter for Celery logs
class CeleryJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        log_record["framework"] = "CELERY FRAMEWORK"
        log_record["federation"] = worker_config.federation
        log_record["worker_role"] = f"exareme2-{worker_config.role.lower()}"
        log_record["worker_identifier"] = worker_config.identifier
        log_record["tag"] = "FRAMEWORK"


@signals.setup_logging.connect
def setup_celery_logging(*args, **kwargs):
    logger = logging.getLogger()
    formatter = CeleryJsonFormatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.setLevel(worker_config.framework_log_level)


app.conf.worker_concurrency = worker_config.celery.worker_concurrency

configure_celery_app_to_use_priority_queue(app)

# Log early broker connection attempt
worker_logger.info("Connecting with broker...")

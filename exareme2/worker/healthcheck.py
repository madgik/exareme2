from exareme2.celery_app_conf import CELERY_APP_QUEUE_MAX_PRIORITY
from exareme2.celery_app_conf import get_celery_app
from exareme2.worker import config as worker_config

HEALTHCHECK_TASK_SIGNATURE = "exareme2.worker.worker_info.worker_info_api.healthcheck"

socket_addr = f"{worker_config.rabbitmq.ip}:{worker_config.rabbitmq.port}"
user = worker_config.rabbitmq.user
password = worker_config.rabbitmq.password
vhost = worker_config.rabbitmq.vhost
celery_app = get_celery_app(user, password, socket_addr, vhost)

healthcheck_signature = celery_app.signature(HEALTHCHECK_TASK_SIGNATURE)
healthcheck_signature.apply_async(
    kwargs={"request_id": "HEALTHCHECK", "check_db": True},
    priority=CELERY_APP_QUEUE_MAX_PRIORITY,
).get(worker_config.celery.tasks_timeout)

print("Healthcheck successful!")

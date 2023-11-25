from exareme2.celery_app_conf import CELERY_APP_QUEUE_MAX_PRIORITY
from exareme2.celery_app_conf import get_celery_app
from exareme2.node import config as node_config

HEALTHCHECK_TASK_SIGNATURE = "exareme2.node.celery_tasks.node_info.healthcheck"

socket_addr = f"{node_config.rabbitmq.ip}:{node_config.rabbitmq.port}"
user = node_config.rabbitmq.user
password = node_config.rabbitmq.password
vhost = node_config.rabbitmq.vhost
celery_app = get_celery_app(user, password, socket_addr, vhost)

healthcheck_signature = celery_app.signature(HEALTHCHECK_TASK_SIGNATURE)
healthcheck_signature.apply_async(
    kwargs={"request_id": "HEALTHCHECK"}, priority=CELERY_APP_QUEUE_MAX_PRIORITY
).get(node_config.celery.tasks_timeout)

print("Healthcheck successful!")

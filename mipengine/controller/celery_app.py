from celery import Celery

from mipengine.controller import config as controller_config

celery_apps = {}


def get_node_celery_app(socket_addr):
    if socket_addr in celery_apps:
        return celery_apps[socket_addr]
        # task = app.signature("mipengine.node.tasks.common.get_node_info")
        # try:
        #     task.delay(request_id="PING")

    user = controller_config.rabbitmq.user
    password = controller_config.rabbitmq.password
    vhost = controller_config.rabbitmq.vhost
    broker = f"pyamqp://{user}:{password}@{socket_addr}/{vhost}"
    broker_transport_options = {
        "max_retries": controller_config.rabbitmq.celery_tasks_max_retries,
        "interval_start": controller_config.rabbitmq.celery_tasks_interval_start,
        "interval_step": controller_config.rabbitmq.celery_tasks_interval_step,
        "interval_max": controller_config.rabbitmq.celery_tasks_interval_max,
    }
    cel_app = Celery(broker=broker, backend="rpc://")
    cel_app.conf.broker_transport_options = broker_transport_options
    cel_app.conf.broker_pool_limit = None

    celery_apps[socket_addr] = cel_app
    return cel_app


def remove_app(socket_addr):
    if socket_addr in celery_apps:
        celery_apps.pop(socket_addr)
        print(celery_apps)

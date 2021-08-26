from celery import Celery

from mipengine.controller import config as controller_config


def get_node_celery_app(socket_addr):
    user = controller_config.rabbitmq.user
    password = controller_config.rabbitmq.password
    vhost = controller_config.rabbitmq.vhost
    broker = f"amqp://{user}:{password}@{socket_addr}/{vhost}"
    broker_transport_options = {
        "max_retries": 3,
        "interval_start": 0,
        "interval_step": 0.2,
        "interval_max": 0.5,
    }
    cel_app = Celery(broker=broker, backend="rpc://")
    cel_app.conf.broker_transport_options = broker_transport_options

    return cel_app

import asyncio

from asgiref.sync import sync_to_async
from celery import Celery

from mipengine.controller import config as controller_config

CELERY_TASKS_TIMEOUT = controller_config.celery_tasks_timeout


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


# Converts a Celery task to an async function
# Celery doesn't currently support asyncio "await" while "getting" a result
# Copied from https://github.com/celery/celery/issues/6603
def task_to_async(task):
    async def wrapper(*args, **kwargs):
        total_delay = 0
        delay = 0.1
        async_result = await sync_to_async(task.delay)(*args, **kwargs)
        while not async_result.ready():
            total_delay += delay
            if total_delay > CELERY_TASKS_TIMEOUT:
                raise TimeoutError(
                    f"Celery task: {task} didn't respond in {CELERY_TASKS_TIMEOUT}s."
                )
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, 2)  # exponential backoff, max 2 seconds
        return async_result.get(timeout=CELERY_TASKS_TIMEOUT - total_delay)

    return wrapper

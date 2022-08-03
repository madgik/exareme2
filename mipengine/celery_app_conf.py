from kombu import Exchange
from kombu import Queue

CELERY_APP_QUEUE_MAX_PRIORITY = 10
CELERY_APP_QUEUE_DEFAULT_PRIORITY = 5
CELERY_APP_DEFAULT_QUEUE_NAME = "celery"


def configure_celery_app_to_use_priority_queue(app):
    # Disable prefetching
    app.conf.worker_prefetch_multiplier = 1
    # task_acks_late is also needed to disable prefetching
    # https://stackoverflow.com/questions/16040039/understanding-celery-task-prefetching/33357180#33357180
    app.conf.task_acks_late = True

    app.conf.task_queue_max_priority = CELERY_APP_QUEUE_MAX_PRIORITY
    app.conf.task_default_priority = CELERY_APP_QUEUE_DEFAULT_PRIORITY
    app.conf.task_queues = [
        Queue(
            CELERY_APP_DEFAULT_QUEUE_NAME,
            Exchange(CELERY_APP_DEFAULT_QUEUE_NAME),
            routing_key=CELERY_APP_DEFAULT_QUEUE_NAME,
            queue_arguments={"x-max-priority": CELERY_APP_QUEUE_MAX_PRIORITY},
        ),
    ]

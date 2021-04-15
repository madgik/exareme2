import argparse

from celery import Celery

from mipengine.node import config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    extra_args, celery_args = parser.parse_known_args()

    rabbitmq_credentials = config.rabbitmq.user + ":" + config.rabbitmq.password
    rabbitmq_url = config.rabbitmq.ip + ":" + config.rabbitmq.port
    vhost = config.rabbitmq.vhost

    app = Celery(
        "mipengine.node",
        broker=f"amqp://{rabbitmq_credentials}@{rabbitmq_url}/{vhost}",
        backend="rpc://",
        include=[
            "mipengine.node.tasks.tables",
            "mipengine.node.tasks.remote_tables",
            "mipengine.node.tasks.merge_tables",
            "mipengine.node.tasks.views",
            "mipengine.node.tasks.common",
            "mipengine.node.tasks.udfs",
        ],
    )

    app.conf.worker_concurrency = config.celery.worker_concurrency
    app.conf.task_soft_time_limit = config.celery.task_soft_time_limit
    app.conf.task_time_limit = config.celery.task_time_limit

    # Start celery
    app.worker_main(celery_args)

from celery import Celery

from mipengine.common.node_catalog import node_catalog
from mipengine.node import config as node_config


def get_celery_app(node_id: str):
    node = node_catalog.get_node(node_id)

    rabbitmq_credentials = (
        node_config.rabbitmq.user + ":" + node_config.rabbitmq.password
    )
    rabbitmq_socket_addr = node.rabbitmqIp + ":" + str(node.rabbitmqPort)
    vhost = node_config.rabbitmq.vhost

    return Celery(
        "mipengine.node",
        broker=f"amqp://{rabbitmq_credentials}@{rabbitmq_socket_addr}/{vhost}",
        backend="rpc://",
        include=[
            "mipengine.node.tasks.tables",
            "mipengine.node.tasks.remote_tables",
            "mipengine.node.tasks.merge_tables",
            "mipengine.node.tasks.views",
            "mipengine.node.tasks.udfs",
            "mipengine.node.tasks.common",
        ],
    )


def get_celery_task_signature(celery_app, task):
    signature_mapping = {
        "create_table": celery_app.signature(
            "mipengine.node.tasks.tables.create_table"
        ),
        "get_tables": celery_app.signature("mipengine.node.tasks.tables.get_tables"),
        "insert_data_to_table": celery_app.signature(
            "mipengine.node.tasks.tables.insert_data_to_table"
        ),
        "create_view": celery_app.signature("mipengine.node.tasks.views.create_view"),
        "create_pathology_view": celery_app.signature(
            "mipengine.node.tasks.views.create_pathology_view"
        ),
        "get_views": celery_app.signature("mipengine.node.tasks.views.get_views"),
        "create_merge_table": celery_app.signature(
            "mipengine.node.tasks.merge_tables.create_merge_table"
        ),
        "get_merge_tables": celery_app.signature(
            "mipengine.node.tasks.merge_tables.get_merge_tables"
        ),
        "create_remote_table": celery_app.signature(
            "mipengine.node.tasks.remote_tables.create_remote_table"
        ),
        "get_remote_tables": celery_app.signature(
            "mipengine.node.tasks.remote_tables.get_remote_tables"
        ),
        "get_table_schema": celery_app.signature(
            "mipengine.node.tasks.common.get_table_schema"
        ),
        "get_table_data": celery_app.signature(
            "mipengine.node.tasks.common.get_table_data"
        ),
        "get_udfs": celery_app.signature("mipengine.node.tasks.udfs.get_udfs"),
        "run_udf": celery_app.signature("mipengine.node.tasks.udfs.run_udf"),
        "get_run_udf_query": celery_app.signature(
            "mipengine.node.tasks.udfs.get_run_udf_query"
        ),
        "clean_up": celery_app.signature("mipengine.node.tasks.common.clean_up"),
    }

    if task not in signature_mapping.keys():
        raise ValueError(f"Task: {task} is not a valid task.")
    return signature_mapping.get(task)

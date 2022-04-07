import sqlalchemy
import toml
from celery import Celery

from mipengine import AttrDict
from tasks import NODES_CONFIG_DIR


def get_node_config_by_id(node_id: str):
    with open(NODES_CONFIG_DIR / f"{node_id}.toml") as fp:
        node_config = AttrDict(toml.load(fp))
    return node_config


def get_celery_app(node_id: str):
    node_config = get_node_config_by_id(node_id)

    rabbitmq_credentials = (
        node_config.rabbitmq.user + ":" + node_config.rabbitmq.password
    )
    rabbitmq_socket_addr = (
        node_config.rabbitmq.ip + ":" + str(node_config.rabbitmq.port)
    )
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
        "get_node_info": celery_app.signature(
            "mipengine.node.tasks.common.get_node_info"
        ),
        "get_node_datasets_per_data_model": celery_app.signature(
            "mipengine.node.tasks.common.get_node_datasets_per_data_model"
        ),
        "get_data_model_cdes": celery_app.signature(
            "mipengine.node.tasks.common.get_data_model_cdes"
        ),
        "create_table": celery_app.signature(
            "mipengine.node.tasks.tables.create_table"
        ),
        "get_tables": celery_app.signature("mipengine.node.tasks.tables.get_tables"),
        "insert_data_to_table": celery_app.signature(
            "mipengine.node.tasks.tables.insert_data_to_table"
        ),
        "create_view": celery_app.signature("mipengine.node.tasks.views.create_view"),
        "create_data_model_view": celery_app.signature(
            "mipengine.node.tasks.views.create_data_model_view"
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
        "get_udf": celery_app.signature("mipengine.node.tasks.udfs.get_udf"),
        "run_udf": celery_app.signature("mipengine.node.tasks.udfs.run_udf"),
        "get_run_udf_query": celery_app.signature(
            "mipengine.node.tasks.udfs.get_run_udf_query"
        ),
        "clean_up": celery_app.signature("mipengine.node.tasks.common.clean_up"),
    }

    if task not in signature_mapping.keys():
        raise ValueError(f"Task: {task} is not a valid task.")
    return signature_mapping.get(task)


def execute_in_db(node_id, query, *args, **kwargs):
    username = "monetdb"
    password = "monetdb"
    port = get_node_config_by_id(node_id)["monetdb"]["port"]
    dbfarm = "db"
    url = f"monetdb://{username}:{password}@localhost:{port}/{dbfarm}:"
    return (
        sqlalchemy.create_engine(url, echo=True)
        .execute(query, *args, **kwargs)
        .fetchone()
    )

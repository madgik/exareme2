from celery import Celery

from mipengine.node import config as node_config

from mipengine.node_registry import (
    NodeRegistryClient,
    Pathologies,
    Pathology,
    NodeRole,
    NodeParams,
    DBParams,
)
from ipaddress import IPv4Address


rabbitmq_credentials = node_config.rabbitmq.user + ":" + node_config.rabbitmq.password
rabbitmq_socket_addr = node_config.rabbitmq.ip + ":" + str(node_config.rabbitmq.port)
vhost = node_config.rabbitmq.vhost

celery = Celery(
    "mipengine.node",
    broker=f"amqp://{rabbitmq_credentials}@{rabbitmq_socket_addr}/{vhost}",
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

celery.conf.worker_concurrency = node_config.celery.worker_concurrency
celery.conf.task_soft_time_limit = node_config.celery.task_soft_time_limit
celery.conf.task_time_limit = node_config.celery.task_time_limit


def _register_node():
    DB_SERVICE_ID_SUFFIX = "_db"
    node_role = (
        NodeRole.LOCALNODE
        if node_config.role == NodeRole.LOCALNODE
        else NodeRole.GLOBALNODE
    )

    db_id = node_config.identifier + DB_SERVICE_ID_SUFFIX

    node_params = NodeParams(
        id=node_config.identifier,
        ip=IPv4Address(node_config.rabbitmq.ip),
        port=node_config.rabbitmq.port,
        role=node_role,
        db_id=db_id,
    )

    db_params = DBParams(
        id=db_id, ip=IPv4Address(node_config.monetdb.ip), port=node_config.monetdb.port
    )

    # TODO we need some mechanism that reads pathologies and datasets form the dbs
    # For now they are hardcoded here
    pathology_dementia = Pathology(
        name="dementia",
        datasets=["edsd", "ppmi", "desd-synthdata", "fake_longitudinal", "demo_data"],
    )
    pathology_mentalhealth = Pathology(name="mentalhealth", datasets=["demo"])
    pathology_tbi = Pathology(name="tbi", datasets=["tbi_demo2"])

    if node_role == NodeRole.LOCALNODE:
        db_params.pathologies = Pathologies(
            pathologies_list=[
                pathology_dementia,
                pathology_mentalhealth,
                pathology_tbi,
            ]
        )

    nrclient_ip = node_config.node_registry.ip
    nrclient_port = node_config.node_registry.port
    nrclient = NodeRegistryClient(
        consul_server_ip=nrclient_ip, consul_server_port=nrclient_port
    )

    nrclient.register_node(node_params, db_params)


_register_node()

from celery import Celery

# from mipengine.common.node_catalog import node_catalog
from mipengine.node import config as node_config

from mipengine.node_registry.node_registry import NodeRegistryClient
from mipengine.common.node_registry_DTOs import NodeRecord, Pathology, NodeRole
from ipaddress import IPv4Address

# ----------- NodeRegistryClient stuff
DB_SERVICE_ID_SUFFIX = "_db"

node_role = (
    NodeRole.LOCALNODE
    if node_config.role == NodeRole.LOCALNODE.name
    else NodeRole.GLOBALNODE
)

node_record = NodeRecord(
    node_id=node_config.identifier,
    node_role=node_role,
    task_queue_ip=IPv4Address(node_config.rabbitmq.ip),
    task_queue_port=node_config.rabbitmq.port,
    db_id=node_config.identifier + DB_SERVICE_ID_SUFFIX,
    db_ip=IPv4Address(node_config.monetdb.ip),
    db_port=node_config.monetdb.port,
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
    node_record.pathologies = [
        pathology_dementia,
        pathology_mentalhealth,
        pathology_tbi,
    ]

# TODO pass consul ip, port from config??
nrcclient = NodeRegistryClient(
    consul_server_ip=IPv4Address("127.0.0.1"), consul_server_port=8500
)
nrcclient.register_node(node_record)
# ----------- END of NodeRegistryClient stuff

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

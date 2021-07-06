import pymonetdb
from pymonetdb import Connection

# from mipengine.common.node_catalog import node_catalog
from mipengine.node_registry.node_registry import (
    NodeRegistryClient,
    Pathologies,
    Pathology,
    NodeRole,
    NodeParams,
    DBParams,
)
from mipengine.node import config as node_config


def get_node_db_connection(node_id: str) -> Connection:
    nrclient = NodeRegistryClient()
    db = nrclient.get_db_by_node_id(node_id)
    connection = pymonetdb.connect(
        hostname=str(db.ip),
        port=db.port,
        username=node_config.monetdb.username,
        password=node_config.monetdb.password,
        database=node_config.monetdb.database,
    )
    return connection

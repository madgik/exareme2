import pymonetdb
from pymonetdb import Connection

from mipengine.common.node_catalog import node_catalog
from mipengine.node import config as node_config


def get_node_db_connection(node_id: str) -> Connection:
    node = node_catalog.get_node(node_id)
    connection = pymonetdb.connect(
        hostname=node.monetdbIp,
        port=node.monetdbPort,
        username=node_config.monetdb.username,
        password=node_config.monetdb.password,
        database=node_config.monetdb.database,
    )
    return connection

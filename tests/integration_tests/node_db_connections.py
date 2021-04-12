import pymonetdb
from pymonetdb import Connection

from mipengine.common.node_catalog import node_catalog
from mipengine import config


def get_node_db_connection(node_id: str) -> Connection:
    global_node = node_catalog.get_global_node()
    if global_node.nodeId == node_id:
        node = global_node
    else:
        node = node_catalog.get_local_node(node_id)

    monetdb_hostname = node.monetdbHostname
    monetdb_port = node.monetdbPort
    print(monetdb_hostname, monetdb_port)
    connection = pymonetdb.connect(
        username=config.monetdb.username,
        port=monetdb_port,
        password=config.monetdb.password,
        hostname=monetdb_hostname,
        database=config.monetdb.database,
    )
    return connection

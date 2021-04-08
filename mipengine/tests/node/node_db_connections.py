import pymonetdb
from pymonetdb import Connection

from mipengine.common.node_catalog import node_catalog
from mipengine.node.config.config_parser import config


def get_node_db_connection(node_id: str) -> Connection:
    global_node = node_catalog.get_global_node()
    if global_node.nodeId == node_id:
        node = global_node
    else:
        node = node_catalog.get_local_node_data(node_id)

    monetdb_hostname = node.monetdbHostname
    monetdb_port = node.monetdbPort
    print(monetdb_hostname, monetdb_port)
    connection = pymonetdb.connect(
        username=config.get("monet_db", "username"),
        port=monetdb_port,
        password=config.get("monet_db", "password"),
        hostname=monetdb_hostname,
        database=config.get("monet_db", "database"),
    )
    return connection

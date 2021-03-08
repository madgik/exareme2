import queue
import threading
import time

import pymonetdb

from mipengine.common.node_catalog import node_catalog
from mipengine.node.config.config_parser import config

connection_queue = queue.Queue()


def __get_connection():
    global_node = node_catalog.get_global_node()
    if global_node.nodeId == config.get("node", "identifier"):
        node = global_node
    else:
        node = node_catalog.get_local_node_data(config.get("node", "identifier"))
    monetdb_hostname = node.monetdbHostname
    monetdb_port = node.monetdbPort
    return pymonetdb.connect(username=config.get("monet_db", "username"),
                             port=monetdb_port,
                             password=config.get("monet_db", "password"),
                             hostname=monetdb_hostname,
                             database=config.get("monet_db", "database"))


for count in range(50):
    connection_queue.put(__get_connection())


def get_connection():
    return connection_queue.get()


def release_connection(connection):
    connection_queue.put(connection)

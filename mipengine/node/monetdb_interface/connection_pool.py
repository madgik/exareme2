import time

import pymonetdb
from pymonetdb import Connection
from pymonetdb.sql.cursors import Cursor

from mipengine.common.node_catalog import node_catalog
from mipengine.node.config.config_parser import config


def get_connection():
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


def execute_with_occ(connection:Connection, cursor: Cursor, query: str):
    attempts = 20
    while attempts >= 0:
        try:
            cursor.execute(query)
            connection.commit()
            break
        except pymonetdb.exceptions.OperationalError as operational_error_exc:
            print(operational_error_exc)
            raise operational_error_exc
        except Exception:
            connection.rollback()
            if attempts == 0:
                raise TimeoutError
            time.sleep(1)
            attempts -= 1


def release_connection(connection, cursor):
    cursor.close()
    connection.close()

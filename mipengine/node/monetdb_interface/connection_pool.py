import time

import pymonetdb
from pymonetdb import Connection
from pymonetdb.sql.cursors import Cursor

from mipengine.common.node_catalog import node_catalog
from mipengine.node.config.config_parser import config


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class MonetDB(metaclass=Singleton):
    """
    MonetDB is a Singleton class because we want it to be initialized at runtime.

    If the connection is a public module variable, it will be initialized at import time
    from Celery and all the Celery workers will use the same connection instance.

    We want one MonetDB connection instance per Celery worker/process.

    """
    def __init__(self):
        print("Initializing MonetDB!")
        global_node = node_catalog.get_global_node()
        if global_node.nodeId == config.get("node", "identifier"):
            node = global_node
        else:
            node = node_catalog.get_local_node_data(config.get("node", "identifier"))
        monetdb_hostname = node.monetdbHostname
        monetdb_port = node.monetdbPort
        self._connection = pymonetdb.connect(username=config.get("monet_db", "username"),
                                             port=monetdb_port,
                                             password=config.get("monet_db", "password"),
                                             hostname=monetdb_hostname,
                                             database=config.get("monet_db", "database"))
        self._cursor = self._connection.cursor()

    def get_connection(self):
        # Commit is needed to get the latest changes.
        self._connection.commit()
        return self._connection

    def get_cursor(self):
        return self._cursor


def get_connection():
    connection = MonetDB().get_connection()
    connection.commit()
    print("Getting MonetDB connection! : " + str(connection))
    return connection


def execute_with_occ(connection: Connection, cursor: Cursor, query: str):
    attempts = 20
    while attempts >= 0:
        try:
            cursor.execute(query)
            connection.commit()
            break
        except pymonetdb.exceptions.OperationalError as operational_error_exc:
            print("Operational Error: " + str(operational_error_exc))
            connection.rollback()
            raise operational_error_exc
        except Exception as exc:
            print("Exception: " + str(exc))
            connection.rollback()
            if attempts == 0:
                raise TimeoutError
            time.sleep(1)
            attempts -= 1

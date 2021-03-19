import time
from typing import List

import pymonetdb

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
        self._connection = self.renew_connection()

    def get_connection(self):
        return self._connection

    def renew_connection(self):
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
                                             database=config.get("monet_db", "database"),
                                             autocommit=True)
        return self._connection


def execute(query: str) -> List:
    cursor = MonetDB().get_connection().cursor()
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    return result


def execute_with_occ(query: str):
    attempts = 0
    max_attemps = 5
    cursor = MonetDB().get_connection().cursor()
    while attempts <= max_attemps:
        try:
            cursor.execute(query)
            print(query)
            break
        except pymonetdb.exceptions.OperationalError as operational_error_exc:
            raise operational_error_exc
        except Exception as exc:
            if str(exc).startswith('40000!COMMIT'):
                print("============================================================================================")
                print(query)
                print(exc)
                if attempts == max_attemps:
                    raise exc
                time.sleep(attempts)
                attempts += 1
                cursor = MonetDB().renew_connection().cursor()
            else:
                print(exc)
                raise exc

from typing import List

import pymonetdb

from mipengine.common.node_catalog import node_catalog
from mipengine.node.config.config_parser import config

OCC_MAX_ATTEMPTS = 50


class Singleton(type):
    """
    Copied from https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python
    """

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
        global_node = node_catalog.get_global_node()
        if global_node.nodeId == config.get("node", "identifier"):
            node = global_node
        else:
            node = node_catalog.get_local_node_data(config.get("node", "identifier"))
        monetdb_hostname = node.monetdbHostname
        monetdb_port = node.monetdbPort
        self._connection = pymonetdb.connect(
            username=config.get("monet_db", "username"),
            port=monetdb_port,
            password=config.get("monet_db", "password"),
            hostname=monetdb_hostname,
            database=config.get("monet_db", "database"),
        )

    def _get_connection(self):
        """
        Commits the connection and then retrieves it so it is up-to-date.
        """
        self._connection.commit()
        return self._connection

    def execute_with_result(self, query: str) -> List:
        """
        Used to execute select queries that return a result.

        Should NOT be used to execute "CREATE, DROP, ALTER, UPDATE, ..." statements.
        """
        cursor = self._get_connection().cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        cursor.close()
        return result

    def execute(self, query: str):
        """
        Executes statements that don't have a result. For example "CREATE,DROP,UPDATE".
        And handles the *Optimistic Concurrency Control by giving each call X attempts
        if they fail with pymonetdb.exceptions.IntegrityError .
        *https://www.monetdb.org/blog/optimistic-concurrency-control
        """
        attempts = 0
        connection = self._get_connection()
        while attempts <= OCC_MAX_ATTEMPTS:
            cursor = connection.cursor()
            try:
                cursor.execute(query)
                connection.commit()
                break
            except pymonetdb.exceptions.IntegrityError as integrity_exc:
                connection.rollback()
                attempts += 1
                if attempts >= OCC_MAX_ATTEMPTS:
                    raise integrity_exc
            except Exception as exc:
                connection.rollback()
                raise exc

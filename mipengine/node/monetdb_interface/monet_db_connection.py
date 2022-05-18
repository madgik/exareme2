import random
import threading
from contextlib import contextmanager
from time import sleep
from typing import List

import pymonetdb
from eventlet.lock import Semaphore

from mipengine.node import config as node_config
from mipengine.node import node_logger as logging

BROKEN_PIPE_MAX_ATTEMPTS = 50
OCC_MAX_ATTEMPTS = 50
INTEGRITY_ERROR_RETRY_INTERVAL = 0.5

eventlet_lock = Semaphore()


class MonetDB:
    """
    MonetDB is a Singleton class because we want it to be initialized at runtime.

    If the connection is a public module variable, it will be initialized at import time
    from Celery and all the Celery workers will use the same connection instance.

    We want one MonetDB connection instance per Celery worker/process.
    """

    def __init__(self):
        self._connection_pool = [self.create_connection() for _ in range(24)]
        self._logger = logging.get_logger()

    def create_connection(self):
        return pymonetdb.connect(
            hostname=node_config.monetdb.ip,
            port=node_config.monetdb.port,
            username=node_config.monetdb.username,
            password=node_config.monetdb.password,
            database=node_config.monetdb.database,
        )

    def _get_connection(self):
        conn = self._connection_pool.pop()
        self._logger.info(f"Got connection {conn}.")
        return conn

    def _release_connection(self, conn):
        self._logger.info(f"Releasing connection {conn}.")
        self._connection_pool.append(conn)

    def _replace_connection(self, conn):
        self._logger.info(f"Replacing connection {conn}.")
        conn.close()
        return self.create_connection()

    @contextmanager
    def cursor(self, _connection):
        broken_pipe_error = None
        for _ in range(BROKEN_PIPE_MAX_ATTEMPTS):
            try:
                # We use a single instance of a connection and by committing before a select query we refresh the state
                # of the connection so that it sees changes from other processes/connections.
                # https://stackoverflow.com/questions/9305669/mysql-python-connection-does-not-see-changes-to-database-made
                # -on-another-connect.
                # self._logger.info(f"Commiting before the cursor")
                # _connection.commit()

                self._logger.info("Getting cursor")
                cur = _connection.cursor()
                self._logger.info(f"Got cursor")
                yield cur
                self._logger.info("Closing cursor")
                cur.close()
                break
            except BrokenPipeError as exc:
                broken_pipe_error = exc
                self._logger.info("Broken pipe error")
                _connection = self._replace_connection(_connection)
                continue
            except Exception as exc:
                self._logger.info(f"{exc}")
                raise exc
        else:
            raise broken_pipe_error

    def execute_and_fetchall(self, query: str, parameters=None, many=False) -> List:
        """
        Used to execute select queries that return a result.
        Should NOT be used to execute "CREATE, DROP, ALTER, UPDATE, ..." statements.

        'many' option to provide the functionality of executemany, all results will be fetched.
        'parameters' option to provide the functionality of bind-parameters.
        """
        conn = self._get_connection()

        self._logger.info(
            f"Query: {query} \n, parameters: {str(parameters)}\n, many: {many}, available_connections={len(self._connection_pool)} {conn=}"
        )

        with self.cursor(conn) as cur:
            self._logger.info(f"Execute with {conn=} {query[0:25]=}")
            cur.executemany(query, parameters) if many else cur.execute(
                query, parameters
            )
            self._logger.info(f"Fetchall with {conn=} {query[0:25]=}")
            result = cur.fetchall()
            self._logger.info(f"Got result {result}.")

        self._release_connection(conn)
        return result

    def execute(self, query: str, parameters=None, many=False):
        """
        Executes statements that don't have a result. For example "CREATE,DROP,UPDATE".
        And handles the *Optimistic Concurrency Control by giving each call X attempts
        if they fail with pymonetdb.exceptions.IntegrityError.
        *https://www.monetdb.org/blog/optimistic-concurrency-control

        'many' option to provide the functionality of executemany.
        'parameters' option to provide the functionality of bind-parameters.
        """
        conn = self._get_connection()

        self._logger.info(
            f"Query: {query} \n, parameters: {str(parameters)}\n, many: {many}, available_connections={len(self._connection_pool)} {conn=}"
        )

        for tries in range(OCC_MAX_ATTEMPTS):
            try:
                with self.cursor(conn) as cur:
                    if "CREATE" in query:
                        try:
                            self._logger.info("Aquiring")
                            eventlet_lock.acquire(timeout=5)
                            self._logger.info(f"Execute with {conn=} {query[0:25]=}")
                            cur.executemany(query, parameters) if many else cur.execute(
                                query, parameters
                            )
                            self._logger.info(f"Committing with {conn=} {query[0:25]=}")
                            conn.commit()
                        finally:
                            self._logger.info("Releasing")
                            eventlet_lock.release()
                    else:
                        self._logger.info(f"Execute with {conn}")
                        cur.executemany(query, parameters) if many else cur.execute(
                            query, parameters
                        )
                        self._logger.info(f"Committing with {conn}")
                        conn.commit()
                self._logger.info("Returning the connection")
                self._release_connection(conn)
                self._logger.info("Returned the connection")
                break
            except pymonetdb.exceptions.IntegrityError as exc:
                integrity_error = exc
                conn.rollback()
                self._logger.error(f"INTEGRITY ERROR: {str(exc)}")
                sleep(INTEGRITY_ERROR_RETRY_INTERVAL)
                continue
            except Exception as exc:
                conn.rollback()
                self._release_connection(conn)
                raise exc
        else:
            self._release_connection(conn)
            raise integrity_error
        self._logger.info(f"{query[0:20]=} DONE!")


monetdb = MonetDB()

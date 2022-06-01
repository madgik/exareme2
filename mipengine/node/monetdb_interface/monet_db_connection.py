import random
import threading
from contextlib import contextmanager
from time import sleep
from typing import List

import pymonetdb
from eventlet.lock import Semaphore

from mipengine.node import config as node_config
from mipengine.node import node_logger as logging
from mipengine.singleton import Singleton

BROKEN_PIPE_MAX_ATTEMPTS = 50
OCC_MAX_ATTEMPTS = 50
INTEGRITY_ERROR_RETRY_INTERVAL = 0.5

create_eventlet_lock = Semaphore()
insert_eventlet_lock = Semaphore()


class DBExecutionDTO:
    def __init__(self, query, parameters=None, many=False):
        self.query = query
        self.parameters = parameters
        self.many = many


class MonetDB(metaclass=Singleton):
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
        return conn

    def _release_connection(self, conn):
        self._connection_pool.append(conn)

    def _replace_connection(self):
        return self.create_connection()

    def execute_and_commit(self, conn, db_execution_dto):
        with self.cursor(conn) as cur:
            cur.executemany(
                db_execution_dto.query, db_execution_dto.parameters
            ) if db_execution_dto.many else cur.execute(
                db_execution_dto.query, db_execution_dto.parameters
            )
        conn.commit()

    @contextmanager
    def cursor(self, _connection):
        # We use a single instance of a connection and by committing before a select query we refresh the state
        # of the connection so that it sees changes from other processes/connections.
        # https://stackoverflow.com/questions/9305669/mysql-python-connection-does-not-see-changes-to-database-made
        # -on-another-connect.
        _connection.commit()

        cur = _connection.cursor()
        yield cur
        cur.close()

    def execute_and_fetchall(self, db_execution_dto: DBExecutionDTO) -> List:
        """
        Used to execute select queries that return a result.
        Should NOT be used to execute "CREATE, DROP, ALTER, UPDATE, ..." statements.

        'many' option to provide the functionality of executemany, all results will be fetched.
        'parameters' option to provide the functionality of bind-parameters.
        """
        conn = self._get_connection()

        self._logger.info(
            f"query: {db_execution_dto.query} \n, parameters: {str(db_execution_dto.parameters)}\n, many: {db_execution_dto.many}"
        )

        for tries in range(OCC_MAX_ATTEMPTS):
            try:
                with self.cursor(conn) as cur:
                    cur.executemany(
                        db_execution_dto.query, db_execution_dto.parameters
                    ) if db_execution_dto.many else cur.execute(
                        db_execution_dto.query, db_execution_dto.parameters
                    )
                    result = cur.fetchall()
                self._release_connection(conn)
                return result
            except BrokenPipeError as exc:
                conn = self._replace_connection()
                sleep(tries * 0.2)
                continue
            except Exception as exc:
                conn.rollback()
                self._release_connection(conn)
                raise exc
        else:
            self._release_connection(conn)
            raise exc

    def execute(self, db_execution_dto):
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
            f"query: {db_execution_dto.query} \n, parameters: {str(db_execution_dto.parameters)}\n, many: {db_execution_dto.many}"
        )

        for tries in range(OCC_MAX_ATTEMPTS):
            try:
                if "CREATE OR REPLACE FUNCTION" in db_execution_dto.query:
                    try:
                        create_eventlet_lock.acquire(timeout=5)
                        self.execute_and_commit(conn, db_execution_dto)
                    finally:
                        create_eventlet_lock.release()
                elif "INSERT INTO" in db_execution_dto.query:
                    try:
                        insert_eventlet_lock.acquire(timeout=5)
                        self.execute_and_commit(conn, db_execution_dto)
                    finally:
                        insert_eventlet_lock.release()
                else:
                    self.execute_and_commit(conn, db_execution_dto)
                self._release_connection(conn)
                break
            except pymonetdb.exceptions.IntegrityError as exc:
                conn.rollback()
                sleep(tries * 0.2)
                continue
            except BrokenPipeError as exc:
                conn = self._replace_connection()
                sleep(tries * 0.2)
                continue
            except Exception as exc:
                conn.rollback()
                self._release_connection(conn)
                raise exc
        else:
            self._release_connection(conn)
            raise exc

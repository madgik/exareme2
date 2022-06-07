from contextlib import contextmanager
from functools import wraps
from time import sleep
from typing import List

import pymonetdb
from eventlet.lock import Semaphore

from mipengine.node import config as node_config
from mipengine.node import node_logger as logging
from mipengine.singleton import Singleton

MAX_ATTEMPTS = 50
BROKEN_PIPE_ERROR_RETRY = 0.2
CREATE_OR_REPLACE_QUERY_TIMEOUT = 1
INSERT_INTO_QUERY_TIMEOUT = (
    node_config.celery.worker_concurrency * node_config.celery.run_udf_time_limit
)

create_function_query_lock = Semaphore()
insert_query_lock = Semaphore()


class DBExecutionDTO:
    def __init__(self, query, parameters=None, many=False):
        self.query = query
        self.parameters = parameters
        self.many = many


def exception_handling(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        for tries in range(MAX_ATTEMPTS):
            conn = self._get_connection()

            try:
                function = func(self, *args, **kwargs, conn=conn)
                break
            except BrokenPipeError as bpe:
                conn = self._create_connection()
                sleep(tries * BROKEN_PIPE_ERROR_RETRY)
                continue
            except Exception as exc:
                conn.rollback()
                raise exc
            finally:
                self._release_connection(conn)
        else:
            raise bpe
        return function

    return wrapper


class MonetDBPool(metaclass=Singleton):
    """
    MonetDBPool is a Singleton class because we want it to be initialized at runtime.

    If the connection is a public module variable, it will be initialized at import time
    from Celery and all the Celery workers will use the same connection instance.

    We want one MonetDB connection instance per Celery worker/process.
    """

    def __init__(self):
        self._connection_pool = [self._create_connection() for _ in range(24)]
        self._logger = logging.get_logger()

    def _create_connection(self):
        return pymonetdb.connect(
            hostname=node_config.monetdb.ip,
            port=node_config.monetdb.port,
            username=node_config.monetdb.username,
            password=node_config.monetdb.password,
            database=node_config.monetdb.database,
        )

    def _get_connection(self):
        return self._connection_pool.pop()

    def _release_connection(self, conn):
        self._connection_pool.append(conn)

    def _execute_and_commit(self, conn, db_execution_dto):
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

    @contextmanager
    def lock(self, query_lock, timeout):
        query_lock.acquire(timeout=timeout)
        yield
        query_lock.release()

    @exception_handling
    def execute_and_fetchall(
        self, query: str, parameters=None, many=False, conn=None
    ) -> List:
        """
        Used to execute select queries that return a result.
        Should NOT be used to execute "CREATE, DROP, ALTER, UPDATE, ..." statements.

        'many' option to provide the functionality of executemany, all results will be fetched.
        'parameters' option to provide the functionality of bind-parameters.
        """
        db_execution_dto = DBExecutionDTO(query=query, parameters=parameters, many=many)

        self._logger.info(
            f"query: {db_execution_dto.query} \n, parameters: {str(db_execution_dto.parameters)}\n, many: {db_execution_dto.many}"
        )

        with self.cursor(conn) as cur:
            cur.executemany(
                db_execution_dto.query, db_execution_dto.parameters
            ) if db_execution_dto.many else cur.execute(
                db_execution_dto.query, db_execution_dto.parameters
            )
            result = cur.fetchall()
        return result

    @exception_handling
    def execute(self, query: str, parameters=None, many=False, conn=None):
        """
        Executes statements that don't have a result. For example "CREATE,DROP,UPDATE".
        And handles the *Optimistic Concurrency Control by giving each call X attempts
        if they fail with pymonetdb.exceptions.IntegrityError.
        *https://www.monetdb.org/blog/optimistic-concurrency-control

        'many' option to provide the functionality of executemany.
        'parameters' option to provide the functionality of bind-parameters.
        """
        db_execution_dto = DBExecutionDTO(query=query, parameters=parameters, many=many)

        self._logger.info(
            f"query: {db_execution_dto.query} \n, parameters: {str(db_execution_dto.parameters)}\n, many: {db_execution_dto.many}"
        )

        if "CREATE OR REPLACE FUNCTION" in db_execution_dto.query:
            with create_function_query_lock:
                self._execute_and_commit(conn, db_execution_dto)
        elif "INSERT INTO" in db_execution_dto.query:
            with insert_query_lock:
                self._execute_and_commit(conn, db_execution_dto)
        else:
            self._execute_and_commit(conn, db_execution_dto)

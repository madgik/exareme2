from contextlib import contextmanager
from typing import List

import pymonetdb
from eventlet.greenthread import sleep
from eventlet.lock import Semaphore

from mipengine.node import config as node_config
from mipengine.node import node_logger as logging
from mipengine.singleton import Singleton

INTEGRITY_ERROR_RETRY_INTERVAL = 1
BROKEN_PIPE_MAX_ATTEMPTS = 50
BROKEN_PIPE_ERROR_RETRY = 0.2
QUERY_EXECUTION_LOCK_TIMEOUT = (
    node_config.celery.worker_concurrency * node_config.celery.run_udf_time_limit
)

query_execution_lock = Semaphore()


class DBExecutionDTO:
    def __init__(self, query, parameters=None, many=False):
        self.query = query
        self.parameters = parameters
        self.many = many


class _MonetDBConnectionPool(metaclass=Singleton):
    """
    MonetDBConnectionPool is a Singleton class.

    We use pseudo-multithreading(eventlet greenlets),
    we provide a connection pool to support concurrent query execution.
    """

    def __init__(self):
        self._connection_pool = [self.create_connection() for _ in range(16)]

    def create_connection(self):
        return pymonetdb.connect(
            hostname=node_config.monetdb.ip,
            port=node_config.monetdb.port,
            username=node_config.monetdb.username,
            password=node_config.monetdb.password,
            database=node_config.monetdb.database,
        )

    def get_connection(self):
        return self._connection_pool.pop()

    def release_connection(self, conn):
        self._connection_pool.append(conn)


@contextmanager
def cursor(_connection):
    """
    We use pseudo-multithreading (eventlet greenlets) by committing before a select query we refresh the state
    of the connection so that it sees changes from other connections.
    https://stackoverflow.com/questions/9305669/mysql-python-connection-does-not-see-changes-to-database-made
    """
    _connection.commit()

    cur = _connection.cursor()
    yield cur
    cur.close()


def _execute_and_commit(conn, db_execution_dto):
    with cursor(conn) as cur:
        cur.executemany(
            db_execution_dto.query, db_execution_dto.parameters
        ) if db_execution_dto.many else cur.execute(
            db_execution_dto.query, db_execution_dto.parameters
        )
    conn.commit()


@contextmanager
def _lock(query_lock, timeout):
    query_lock.acquire(timeout=timeout)
    yield
    query_lock.release()


def db_execute_and_fetchall(query: str, parameters=None, many=False) -> List:
    return execute_queries_with_connection_handling(
        func=_execute_and_fetchall,
        query=query,
        parameters=parameters,
        many=many,
    )


def db_execute(query: str, parameters=None, many=False) -> List:

    return execute_queries_with_connection_handling(
        func=_execute, query=query, parameters=parameters, many=many
    )


def execute_queries_with_connection_handling(func, *args, **kwargs):
    """
    On the query execution we need to handle the 'BrokenPipeError' exception.
    In the case of the 'BrokenPipeError' exception, we create a new connection,
    and we retry for x amount of times the execution in case the database has recovered.
    """
    for tries in range(BROKEN_PIPE_MAX_ATTEMPTS):
        conn = _MonetDBConnectionPool().get_connection()

        try:
            return func(*args, **kwargs, conn=conn)
        except BrokenPipeError as exc:
            conn = _MonetDBConnectionPool().create_connection()
            sleep(tries * BROKEN_PIPE_ERROR_RETRY)
            continue
        except Exception as exc:
            conn.rollback()
            raise exc
        finally:
            _MonetDBConnectionPool().release_connection(conn)
    else:
        raise exc


def _execute_and_fetchall(query: str, parameters=None, many=False, conn=None) -> List:
    """
    Used to execute only select queries that return a result.

    'many' option to provide the functionality of executemany, all results will be fetched.
    'parameters' option to provide the functionality of bind-parameters.
    """
    logger = logging.get_logger()
    db_execution_dto = DBExecutionDTO(query=query, parameters=parameters, many=many)
    logger.info(
        f"query: {db_execution_dto.query} \n, parameters: {str(db_execution_dto.parameters)}\n, many: {db_execution_dto.many}"
    )

    with cursor(conn) as cur:
        cur.executemany(
            db_execution_dto.query, db_execution_dto.parameters
        ) if db_execution_dto.many else cur.execute(
            db_execution_dto.query, db_execution_dto.parameters
        )
        result = cur.fetchall()
    return result


def _execute(query: str, parameters=None, many=False, conn=None):
    """
    Executes statements that don't have a result. For example "CREATE,DROP,UPDATE".

    By adding create_function_query_lock we serialized the execution of the queries that contain 'create remote table',
    in order to a bug that was found.
    https://github.com/MonetDB/MonetDB/issues/7304

    By adding create_remote_table_query_lock we serialized the execution of the queries that contain 'create or replace function',
    in order to handle the error 'CREATE OR REPLACE FUNCTION: transaction conflict detected'
    https://www.mail-archive.com/checkin-list@monetdb.org/msg46062.html

    By adding insert_query_lock we serialized the execution of the queries that contain 'INSERT INTO'.
    We need this insert_query_lock in order to ensure that we will have the zero-cost that the monetdb provides on the udfs.

    'many' option to provide the functionality of executemany.
    'parameters' option to provide the functionality of bind-parameters.
    """
    logger = logging.get_logger()
    db_execution_dto = DBExecutionDTO(query=query, parameters=parameters, many=many)

    logger.info(
        f"query: {db_execution_dto.query} \n, parameters: {str(db_execution_dto.parameters)}\n, many: {db_execution_dto.many}"
    )

    if "CREATE" or "INSERT" or "DROP" in db_execution_dto.query:
        with _lock(query_execution_lock, QUERY_EXECUTION_LOCK_TIMEOUT):
            _execute_and_commit(conn, db_execution_dto)
    else:
        _execute_and_commit(conn, db_execution_dto)

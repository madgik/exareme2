from contextlib import contextmanager
from typing import List

import pymonetdb
from eventlet.greenthread import sleep
from eventlet.lock import Semaphore

from mipengine.node import config as node_config
from mipengine.node import node_logger as logging
from mipengine.singleton import Singleton

INTEGRITY_ERROR_RETRY_INTERVAL = 1
CONN_RECOVERY_MAX_ATTEMPTS = 10
CONN_RECOVERY_ERROR_RETRY = 0.2
QUERY_EXECUTION_LOCK_TIMEOUT = node_config.celery.tasks_timeout
UDF_EXECUTION_LOCK_TIMEOUT = node_config.celery.run_udf_task_timeout

query_execution_lock = Semaphore()
udf_execution_lock = Semaphore()


class DBExecutionDTO:
    def __init__(self, query, parameters=None, many=False):
        self.query = query
        self.parameters = parameters
        self.many = many


class LockDTO:
    def __init__(self, lock, timeout):
        self.lock = lock
        self.timeout = timeout


class _MonetDBConnectionPool(metaclass=Singleton):
    """
    MonetDBConnectionPool is a Singleton class.

    We use pseudo-multithreading(eventlet greenlets),
    we provide a connection pool to support concurrent query execution.
    """

    def __init__(self):
        self._connection_pool = []

    def create_connection(self):
        return pymonetdb.connect(
            hostname=node_config.monetdb.ip,
            port=node_config.monetdb.port,
            username=node_config.monetdb.username,
            password=node_config.monetdb.password,
            database=node_config.monetdb.database,
        )

    def get_connection(self):
        """
        We use pseudo-multithreading (eventlet greenlets) by committing before a query we refresh the state
        of the connection so that it sees changes from other connections.
        https://stackoverflow.com/questions/9305669/mysql-python-connection-does-not-see-changes-to-database-made
        """
        if not self._connection_pool:
            connection = self.create_connection()
        else:
            connection = self._connection_pool.pop()
        connection.commit()
        return connection

    def release_connection(self, connection):
        connection.commit()
        self._connection_pool.append(connection)


@contextmanager
def cursor(_connection):
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
    acquired = query_lock.acquire(timeout=timeout)
    if not acquired:
        raise TimeoutError()
    try:
        yield
    finally:
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
    On the query execution we need to handle the 'OSError' and 'BrokenPipeError' exception.
    On both cases we try for x amount of times to recover the connection with the database.
    """
    conn = None
    for tries in range(CONN_RECOVERY_MAX_ATTEMPTS):

        try:
            conn = _MonetDBConnectionPool().get_connection()
            result = func(*args, **kwargs, conn=conn)
            _MonetDBConnectionPool().release_connection(conn)
            return result
        except (BrokenPipeError, OSError) as exc:
            logger = logging.get_logger()
            logger.warning(
                f"Trying to recover the connection with the database. Exception type: '{type(exc)}', exc: '{exc}'"
            )
            sleep(tries * CONN_RECOVERY_ERROR_RETRY)
        except Exception as exc:
            if conn:
                conn.rollback()
                _MonetDBConnectionPool().release_connection(conn)
            raise exc
    else:
        raise ConnectionError("Failed to connect with the database.")


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

    lock_dto = (
        LockDTO(udf_execution_lock, UDF_EXECUTION_LOCK_TIMEOUT)
        if db_execution_dto.query.startswith("INSERT INTO")
        else LockDTO(query_execution_lock, QUERY_EXECUTION_LOCK_TIMEOUT)
    )

    try:
        with _lock(lock_dto.lock, lock_dto.timeout):
            _execute_and_commit(conn, db_execution_dto)
    except TimeoutError:
        error_msg = _get_lock_timeout_error_msg(db_execution_dto, lock_dto.timeout)
        raise TimeoutError(error_msg)


def _get_lock_timeout_error_msg(db_execution_dto, lock_timeout):
    return f"""
        query: {db_execution_dto.query=} with parameters:
        {str(db_execution_dto.parameters)} and
        {db_execution_dto.many=} was not executed because the
        lock was not acquired during
        {lock_timeout=}
    """

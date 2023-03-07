from contextlib import contextmanager
from functools import wraps
from typing import Any
from typing import List
from typing import Optional

import pymonetdb
from eventlet.greenthread import sleep
from eventlet.lock import Semaphore
from pydantic import BaseModel
from pymonetdb import ProgrammingError

from mipengine.node import config as node_config
from mipengine.node import node_logger as logging

UDF_EXECUTION_QUERY_PREFIX = "INSERT INTO"
query_execution_lock = Semaphore()
udf_execution_lock = Semaphore()


class _DBExecutionDTO(BaseModel):
    query: str
    parameters: Optional[List[Any]]
    timeout: Optional[int]

    class Config:
        allow_mutation = False


def db_execute_and_fetchall(query: str, parameters=None) -> List:
    query_execution_timeout = node_config.celery.tasks_timeout
    db_execution_dto = _DBExecutionDTO(
        query=query, parameters=parameters, timeout=query_execution_timeout
    )
    return _execute_and_fetchall(db_execution_dto=db_execution_dto)


# TODO:https://team-1617704806227.atlassian.net/browse/MIP-763
def db_execute(query: str, parameters=None):
    if query.startswith(UDF_EXECUTION_QUERY_PREFIX):
        _db_execute_udf(query, parameters)
    else:
        _db_execute_query(query, parameters)


def _db_execute_query(query: str, parameters=None):
    query_execution_timeout = node_config.celery.run_udf_task_timeout
    db_execution_dto = _DBExecutionDTO(
        query=query, parameters=parameters, timeout=query_execution_timeout
    )
    _execute(db_execution_dto=db_execution_dto, lock=query_execution_lock)


def _db_execute_udf(query: str, parameters=None):
    # Check if there is only one query
    split_queries = [query for query in query.strip().split(";") if query]
    if len(split_queries) > 1:
        raise ValueError(f"UDF execution query: {query} should contain only one query.")

    udf_execution_timeout = node_config.celery.run_udf_task_timeout
    db_execution_dto = _DBExecutionDTO(
        query=query, parameters=parameters, timeout=udf_execution_timeout
    )
    _execute(db_execution_dto=db_execution_dto, lock=udf_execution_lock)


@contextmanager
def _connection():
    conn = pymonetdb.connect(
        hostname=node_config.monetdb.ip,
        port=node_config.monetdb.port,
        username=node_config.monetdb.username,
        password=node_config.monetdb.password,
        database=node_config.monetdb.database,
    )
    yield conn
    conn.close()


@contextmanager
def _cursor(commit=False):
    with _connection() as conn:
        cur = conn.cursor()
        yield cur
        cur.close()
        if commit:
            conn.commit()


@contextmanager
def _lock(query_lock, timeout):
    acquired = query_lock.acquire(timeout=timeout)
    if not acquired:
        raise TimeoutError("Could not acquire the lock in the designed timeout.")
    try:
        yield
    finally:
        query_lock.release()


def _get_table_name_on_query(query: str, query_prefix: str):
    # We need to extract the table name from the query
    string_before_table_name_pos = query.find(query_prefix)
    table_name, *_ = query[string_before_table_name_pos + len(query_prefix) :].split()
    return table_name


def _create_idempotent_insert_into_query(query):
    """
    In order for an insert query to be idempotent,
    we need to delete all the existing values of the table,
    before inserting the new values.'
    """
    if UDF_EXECUTION_QUERY_PREFIX in query:
        table_name = _get_table_name_on_query(query, UDF_EXECUTION_QUERY_PREFIX)
        return f"DELETE FROM {table_name};" + query
    return query


def _create_idempotent_create_merge_table_query(query):
    """
    The creation of the merge table cannot be easily idempotent because adding
    an "IF NOT EXISTS" will leave the added tables. This means that adding afterwards tables
    into the merge table will throw errors. We cannot add an "IF NOT EXISTS" clause in the
    "ADD TABLE TO MERGE TABLE" query because it doesn't exist.
    So the approach we are taking is dropping the merge table each time and recreating it to
    make the whole merge table creation idempotent.
    """
    prefix = "CREATE MERGE TABLE"
    if prefix in query:
        table_name = _get_table_name_on_query(query, prefix)
        return f"DROP TABLE {table_name};" + query
    return query


def _create_idempotent_query(query: str) -> str:
    """
    This is a query optimization method to protect from the following edge case:
    1) A udf starts running allocating memory,
    2) a table creation query starts running,
    3) the udf allocates more memory than monetdb can provide,
    4) the container memory limit kills monetdb,
    5) the table creation query returns with BrokenPipeError("Server closed the connection"),
    6) when the monet_db_facade tries to rerun the failed query it receives a "Table already exists error"
        because the table was created even though the connection was severed.

    The solution to this is to make all queries idempotent.
    """
    idempotent_query = query

    if UDF_EXECUTION_QUERY_PREFIX in idempotent_query:
        idempotent_split_queries = [
            _create_idempotent_insert_into_query(query)
            for query in idempotent_query.strip().split(";")
            if query
        ]
        idempotent_query = ";".join(idempotent_split_queries) + ";"

    if "CREATE MERGE TABLE" in idempotent_query:
        idempotent_split_queries = [
            _create_idempotent_create_merge_table_query(query)
            for query in idempotent_query.strip().split(";")
            if query
        ]
        idempotent_query = ";".join(idempotent_split_queries) + ";"

    if "CREATE" in idempotent_query:
        idempotent_query = idempotent_query.replace(
            "CREATE TABLE", "CREATE TABLE IF NOT EXISTS"
        )
        idempotent_query = idempotent_query.replace(
            "CREATE REMOTE TABLE", "CREATE REMOTE TABLE IF NOT EXISTS"
        )
        idempotent_query = idempotent_query.replace(
            "CREATE VIEW", "CREATE OR REPLACE VIEW"
        )
        idempotent_query = idempotent_query.replace(
            "CREATE FUNCTION", "CREATE OR REPLACE FUNCTION"
        )

    if "DROP" in idempotent_query:
        idempotent_query = idempotent_query.replace(
            "DROP FUNCTION", "DROP FUNCTION IF EXISTS"
        )
        idempotent_query = idempotent_query.replace(
            "DROP TABLE", "DROP TABLE IF EXISTS"
        )
        idempotent_query = idempotent_query.replace("DROP VIEW", "DROP VIEW IF EXISTS")
    return idempotent_query


def _execute_queries_with_error_handling(func):
    @wraps(func)
    def error_handling(**kwargs):
        """
        On the query execution we need to handle the 'BrokenPipeError' and 'pymonetdb.exceptions.DatabaseError' exceptions.
        In these cases we try to recover the connection with the database for x amount of time (x should not exceed the timeout).
        """

        db_execution_dto = kwargs["db_execution_dto"]
        idempotent_query = _create_idempotent_query(db_execution_dto.query)
        idempotent_db_execution_dto = _DBExecutionDTO(
            query=idempotent_query,
            parameters=db_execution_dto.parameters,
            timeout=db_execution_dto.timeout,
        )
        kwargs["db_execution_dto"] = idempotent_db_execution_dto

        logger = logging.get_logger()
        logger.debug(
            f"query: {idempotent_db_execution_dto.query=} \n, parameters: {idempotent_db_execution_dto.parameters}"
        )

        attempts = 0
        # The NODE tasks have a timeout, so there is no point of trying a longer period of time,
        # the task will have timed out in the CONTROLLER.
        # pow(2, attempts + 1) ~= the total amount of time sleeping until this point
        while idempotent_db_execution_dto.timeout > pow(2, attempts + 1):
            try:
                return func(**kwargs)
            except ProgrammingError as exc:
                logger.error(
                    f"Error occurred: Exception type: '{type(exc)}' and exception message: '{exc}'"
                )
                raise exc
            except (BrokenPipeError, pymonetdb.exceptions.DatabaseError) as exc:
                if isinstance(exc, ProgrammingError) and "3F000!" in exc:
                    logger.error(
                        f"Error occurred: Exception type: '{type(exc)}' and exception message: '{exc}'"
                    )
                    raise exc
                logger.warning(
                    f"Trying to recover the connection with the database."
                    f"Exception type: '{type(exc)}' and exception message: '{exc}'."
                    f"Attempts={attempts}"
                )
                # To avoid the flooding of the request to the monetdb when a 'connection error' occur.
                # We retry with a larger interval each time.
                sleep(pow(2, attempts))
                attempts += 1
            except Exception as exc:
                logger.error(
                    f"Error occurred: Exception type: '{type(exc)}' and exception message: '{exc}'"
                )
                raise exc

        connection_error_message = f"Failed to connect with the database. {attempts=}"
        logger.error(connection_error_message)
        raise ConnectionError(connection_error_message)

    return error_handling


@_execute_queries_with_error_handling
def _execute_and_fetchall(db_execution_dto) -> List:
    """
    Used to execute only select queries that return a result.
    'parameters' option to provide the functionality of bind-parameters.
    """
    with _cursor() as cur:
        cur.execute(db_execution_dto.query, db_execution_dto.parameters)
        result = cur.fetchall()
    return result


@_execute_queries_with_error_handling
def _execute(db_execution_dto: _DBExecutionDTO, lock):
    """
    Executes statements that don't have a result. For example "CREATE,DROP,UPDATE,INSERT".

    By adding create_function_query_lock we serialized the execution of the queries that contain 'create remote table',
    in order to a bug that was found.
    https://github.com/MonetDB/MonetDB/issues/7304

    By adding create_remote_table_query_lock we serialized the execution of the queries that contain 'create or replace function',
    in order to handle the error 'CREATE OR REPLACE FUNCTION: transaction conflict detected'
    https://www.mail-archive.com/checkin-list@monetdb.org/msg46062.html

    By adding insert_query_lock we serialized the execution of the queries that contain 'INSERT INTO'.
    We need this insert_query_lock in order to ensure that we will have the zero-cost that the monetdb provides on the udfs.

    'parameters' option to provide the functionality of bind-parameters.
    """

    try:
        with _lock(lock, db_execution_dto.timeout):
            with _cursor(commit=True) as cur:
                cur.execute(db_execution_dto.query, db_execution_dto.parameters)
    except TimeoutError:
        error_msg = f"""
        The execution of {db_execution_dto} failed because the
        lock was not acquired during
        {db_execution_dto.timeout}
        """
        raise TimeoutError(error_msg)


# Connection Pool disabled due to bugs in maintaining connections
# class _MonetDBConnectionPool(metaclass=Singleton):
#     """
#     MonetDBConnectionPool is a Singleton class.
#
#     We use pseudo-multithreading(eventlet greenlets),
#     we provide a connection pool to support concurrent query execution.
#     """
#
#     def __init__(self):
#         self._connection_pool = []
#
#     def create_connection(self):
#         return pymonetdb.connect(
#             hostname=node_config.monetdb.ip,
#             port=node_config.monetdb.port,
#             username=node_config.monetdb.username,
#             password=node_config.monetdb.password,
#             database=node_config.monetdb.database,
#         )
#
#     def clear(self):
#         self._connection_pool.clear()
#
#     def get_connection(self):
#         """
#         We use pseudo-multithreading (eventlet greenlets) by committing before a query we refresh the state
#         of the connection so that it sees changes from other connections.
#         https://stackoverflow.com/questions/9305669/mysql-python-connection-does-not-see-changes-to-database-made
#         """
#         if not self._connection_pool:
#             connection = self.create_connection()
#         else:
#             connection = self._connection_pool.pop()
#         connection.commit()
#
#         return connection
#
#     def release_connection(self, connection):
#         connection.commit()
#         self._connection_pool.append(connection)
#

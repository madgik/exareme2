from contextlib import contextmanager
from functools import wraps
from math import log2
from typing import Any
from typing import List
from typing import Optional

import pymonetdb
from eventlet.greenthread import sleep
from eventlet.lock import Semaphore
from pydantic import BaseModel
from pymonetdb import DatabaseError
from pymonetdb import ProgrammingError

from mipengine.node import config as node_config
from mipengine.node import node_logger as logging

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


def db_execute_query(query: str, parameters=None):
    query_execution_timeout = node_config.celery.tasks_timeout
    db_execution_dto = _DBExecutionDTO(
        query=query, parameters=parameters, timeout=query_execution_timeout
    )
    _execute(db_execution_dto=db_execution_dto, lock=query_execution_lock)


def db_execute_udf(query: str, parameters=None):
    # Check if there is only one query
    split_queries = [query for query in query.strip().split(";") if query]
    if len(split_queries) > 1:
        raise ValueError(f"UDF execution query: {query} should contain only one query.")

    udf_execution_timeout = node_config.celery.run_udf_task_timeout
    db_execution_dto = _DBExecutionDTO(
        query=query, parameters=parameters, timeout=udf_execution_timeout
    )
    _execute(db_execution_dto=db_execution_dto, lock=udf_execution_lock)


# Connection Pool disabled due to bugs in maintaining connections
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


def _validate_exception_is_recoverable(exc):
    """
    Check whether the query needs to be re-executed and return True or False accordingly.
    """
    if isinstance(exc, BrokenPipeError):
        return True
    elif isinstance(exc, DatabaseError):
        return "ValueError" not in str(exc) and not isinstance(exc, ProgrammingError)
    else:
        return False


def _execute_queries_with_error_handling(func):
    @wraps(func)
    def error_handling(**kwargs):
        """
        On the query execution, handle the 'BrokenPipeError' and 'pymonetdb.exceptions.DatabaseError' exceptions.
        In these cases, try to recover the connection with the database for x amount of time (x should not exceed the timeout).
        """
        db_execution_dto = kwargs["db_execution_dto"]

        logger = logging.get_logger()
        logger.debug(
            f"query: {db_execution_dto.query} \n, parameters: {db_execution_dto.parameters}"
        )

        attempts = 0
        max_attempts = int(log2(db_execution_dto.timeout))

        while True:
            try:
                return func(**kwargs)
            except Exception as exc:
                if not _validate_exception_is_recoverable(exc):
                    logger.error(
                        f"Error occurred: Exception type: '{type(exc)}' and exception message: '{exc}'"
                    )
                    raise exc

                logger.warning(
                    f"Trying to recover the connection with the database. "
                    f"Exception type: '{type(exc)}' and exception message: '{exc}'. "
                    f"Attempts={attempts}"
                )
                sleep(pow(2, attempts))
                attempts += 1

                if attempts >= max_attempts:
                    raise exc

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
                query = db_execution_dto.query.replace(
                    "CREATE TABLE", "CREATE TABLE IF NOT EXISTS"
                )
                cur.execute(query, db_execution_dto.parameters)
    except TimeoutError:
        error_msg = f"""
        The execution of {db_execution_dto} failed because the
        lock was not acquired during
        {db_execution_dto.timeout}
        """
        raise TimeoutError(error_msg)

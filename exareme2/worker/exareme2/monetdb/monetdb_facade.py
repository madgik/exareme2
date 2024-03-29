import re
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

from exareme2.worker import config as worker_config
from exareme2.worker.utils import logger as logging

query_execution_lock = Semaphore()
udf_execution_lock = Semaphore()


class _DBExecutionDTO(BaseModel):
    query: str
    parameters: Optional[List[Any]]
    use_public_user: bool = False
    timeout: Optional[int]

    class Config:
        allow_mutation = False


def db_execute_and_fetchall(
    query: str, parameters=None, use_public_user: bool = False
) -> List:
    query_execution_timeout = worker_config.celery.tasks_timeout
    db_execution_dto = _DBExecutionDTO(
        query=query,
        parameters=parameters,
        use_public_user=use_public_user,
        timeout=query_execution_timeout,
    )
    return _execute_and_fetchall(db_execution_dto=db_execution_dto)


def db_execute_query(query: str, parameters=None, use_public_user: bool = False):
    query_execution_timeout = worker_config.celery.tasks_timeout
    query = convert_to_idempotent(query)
    db_execution_dto = _DBExecutionDTO(
        query=query,
        parameters=parameters,
        use_public_user=use_public_user,
        timeout=query_execution_timeout,
    )
    _execute(db_execution_dto=db_execution_dto, lock=query_execution_lock)


def db_execute_udf(query: str, parameters=None):
    # Check if there is only one query
    split_queries = [query for query in query.strip().split(";") if query]
    if len(split_queries) > 1:
        raise ValueError(f"UDF execution query: {query} should contain only one query.")

    udf_execution_timeout = worker_config.celery.run_udf_task_timeout
    query = convert_udf_execution_query_to_idempotent(query)
    db_execution_dto = _DBExecutionDTO(
        query=query, parameters=parameters, timeout=udf_execution_timeout
    )
    _execute(db_execution_dto=db_execution_dto, lock=udf_execution_lock)


# Connection Pool disabled due to bugs in maintaining connections
@contextmanager
def _connection(use_public_user: bool):
    if use_public_user:
        username = worker_config.monetdb.public_username
        password = worker_config.monetdb.public_password
    else:
        username = worker_config.monetdb.local_username
        password = worker_config.monetdb.local_password

    conn = pymonetdb.connect(
        hostname=worker_config.monetdb.ip,
        port=worker_config.monetdb.port,
        username=username,
        password=password,
        database=worker_config.monetdb.database,
    )
    yield conn
    conn.close()


@contextmanager
def _cursor(use_public_user: bool, commit: bool = False):
    with _connection(use_public_user) as conn:
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


def _validate_exception_could_be_recovered(exc):
    """
    Check the error message and decide if this is an exception that could be successful if re-executed.
    ValueError: Cannot be recovered since it's thrown from the udfs.
    InsufficientPrivileges: Cannot be recovered since the user provided doesn't have access.
    ProgrammingError: Cannot be recovered due to udf error.
    """
    if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
        return True
    elif isinstance(exc, DatabaseError):
        if "ValueError" in str(exc):
            return False
        elif "insufficient privileges" in str(exc):
            return False
        elif isinstance(exc, ProgrammingError):
            return False
        return True
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
                if not _validate_exception_could_be_recovered(exc):
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
    with _cursor(use_public_user=db_execution_dto.use_public_user) as cur:
        cur.execute(db_execution_dto.query, db_execution_dto.parameters)
        result = cur.fetchall()
    return result


def convert_udf_execution_query_to_idempotent(query: str) -> str:
    def extract_table_name(query: str) -> str:
        """
        Extracts the name of the table from an INSERT INTO statement.

        Args:
            query (str): The SQL query to extract the table name from.

        Returns:
            str: The name of the table.
        """
        # Use a regular expression to extract the table name
        insert_regex = r"(?i)INSERT\s+INTO\s+(\w+)"
        match = re.search(insert_regex, query)
        if match:
            table_name = match.group(1)
            return table_name
        else:
            raise ValueError("Query is not a valid INSERT INTO statement.")

    return (
        f"{query.rstrip(';')}\n"
        f"WHERE NOT EXISTS (SELECT * FROM {extract_table_name(query)});"
    )


def convert_to_idempotent(query: str) -> str:
    """
    This function creates an idempotent query to protect from a potential edge case
    where a table creation query is interrupted due to a UDF running and allocating memory.
    """
    idempotent_query = query

    if "CREATE" in query:
        idempotent_query = idempotent_query.replace(
            "CREATE TABLE", "CREATE TABLE IF NOT EXISTS"
        )
        idempotent_query = idempotent_query.replace(
            "CREATE MERGE TABLE", "CREATE MERGE TABLE IF NOT EXISTS"
        )
        idempotent_query = idempotent_query.replace(
            "CREATE REMOTE TABLE", "CREATE REMOTE TABLE IF NOT EXISTS"
        )
        idempotent_query = idempotent_query.replace(
            "CREATE VIEW", "CREATE OR REPLACE VIEW"
        )

    if "DROP" in query:
        idempotent_query = idempotent_query.replace(
            "DROP TABLE", "DROP TABLE IF EXISTS"
        )
        idempotent_query = idempotent_query.replace("DROP VIEW", "DROP VIEW IF EXISTS")
        idempotent_query = idempotent_query.replace(
            "DROP FUNCTION", "DROP FUNCTION IF EXISTS"
        )

    return idempotent_query


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
            with _cursor(
                use_public_user=db_execution_dto.use_public_user,
                commit=True,
            ) as cur:
                cur.execute(db_execution_dto.query, db_execution_dto.parameters)
    except TimeoutError:
        error_msg = f"""
        The execution of {db_execution_dto} failed because the
        lock was not acquired during
        {db_execution_dto.timeout}
        """
        raise TimeoutError(error_msg)

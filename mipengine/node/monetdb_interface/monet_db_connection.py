from contextlib import contextmanager
from functools import wraps
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
CREATE_OR_REPLACE_QUERY_TIMEOUT = 1
CREATE_REMOTE_TABLE_QUERY_TIMEOUT = 1
INSERT_INTO_QUERY_TIMEOUT = (
    node_config.celery.worker_concurrency * node_config.celery.run_udf_time_limit
)

create_remote_table_query_lock = Semaphore()
create_function_query_lock = Semaphore()
insert_query_lock = Semaphore()


class DBExecutionDTO:
    def __init__(self, query, parameters=None, many=False):
        self.query = query
        self.parameters = parameters
        self.many = many


def query_execution_exception_handling(func):
    """
    On the query execution we need to handle the 'BrokenPipeError' exception.
    In the case of the 'BrokenPipeError' exception, we create a new connection,
    and we retry for x amount of times the execution in case the database has recovered.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        for tries in range(BROKEN_PIPE_MAX_ATTEMPTS):
            conn = self._get_connection()

            try:
                function = func(self, *args, **kwargs, conn=conn)
                break
            except BrokenPipeError as exc:
                conn = self._create_connection()
                sleep(tries * BROKEN_PIPE_ERROR_RETRY)
                continue
            except Exception as exc:
                conn.rollback()
                raise exc
            finally:
                self._release_connection(conn)
        else:
            raise exc

        return function

    return wrapper


class MonetDBPool(metaclass=Singleton):
    """
    MonetDBPool is a Singleton class because we want it to be initialized at runtime.

    We use sudo-multithreading(eventlet greenlets),
    we provide a connection pool to support concurrent query execution.
    """

    def __init__(self):
        self._logger = logging.get_logger()
        self._connection_pool = [self._create_connection() for _ in range(16)]

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
        """
        We use sudo-multithreading (eventlet greenlets) by committing before a select query we refresh the state
        of the connection so that it sees changes from other connections.
        https://stackoverflow.com/questions/9305669/mysql-python-connection-does-not-see-changes-to-database-made
        """
        _connection.commit()

        cur = _connection.cursor()
        yield cur
        cur.close()

    @contextmanager
    def lock(self, query_lock, timeout):
        query_lock.acquire(timeout=timeout)
        yield
        query_lock.release()

    @query_execution_exception_handling
    def execute_and_fetchall(
        self, query: str, parameters=None, many=False, conn=None
    ) -> List:
        """
        Used to execute only select queries that return a result.

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

    @query_execution_exception_handling
    def execute(self, query: str, parameters=None, many=False, conn=None):
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
        db_execution_dto = DBExecutionDTO(query=query, parameters=parameters, many=many)

        self._logger.info(
            f"query: {db_execution_dto.query} \n, parameters: {str(db_execution_dto.parameters)}\n, many: {db_execution_dto.many}"
        )

        if "CREATE OR REPLACE FUNCTION" in db_execution_dto.query:
            with self.lock(create_function_query_lock, CREATE_OR_REPLACE_QUERY_TIMEOUT):
                self._execute_and_commit(conn, db_execution_dto)
        elif "CREATE REMOTE" in db_execution_dto.query:
            with self.lock(
                create_remote_table_query_lock, CREATE_REMOTE_TABLE_QUERY_TIMEOUT
            ):
                self._execute_and_commit(conn, db_execution_dto)
        elif "INSERT INTO" in db_execution_dto.query:
            with self.lock(insert_query_lock, INSERT_INTO_QUERY_TIMEOUT):
                self._execute_and_commit(conn, db_execution_dto)
        else:
            self._execute_and_commit(conn, db_execution_dto)

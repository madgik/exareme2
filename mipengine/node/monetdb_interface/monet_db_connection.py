from contextlib import contextmanager
from time import sleep
from typing import List

import pymonetdb

from mipengine.node import config as node_config
from mipengine.node import node_logger as logging

BROKEN_PIPE_MAX_ATTEMPTS = 50
OCC_MAX_ATTEMPTS = 50
INTEGRITY_ERROR_RETRY_INTERVAL = 0.5


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
        self._connection = None
        self.refresh_connection()
        self._logger = logging.get_logger()

    def refresh_connection(self):
        self._connection = pymonetdb.connect(
            hostname=node_config.monetdb.ip,
            port=node_config.monetdb.port,
            username=node_config.monetdb.username,
            password=node_config.monetdb.password,
            database=node_config.monetdb.database,
        )

    @contextmanager
    def cursor(self):

        broken_pipe_error = None
        for _ in range(BROKEN_PIPE_MAX_ATTEMPTS):
            try:
                # We use a single instance of a connection and by committing before a select query we refresh the state
                # of the connection so that it sees changes from other processes/connections.
                # https://stackoverflow.com/questions/9305669/mysql-python-connection-does-not-see-changes-to-database-made
                # -on-another-connect.
                self._connection.commit()

                cur = self._connection.cursor()
                yield cur
                cur.close()
                break
            except BrokenPipeError as exc:
                broken_pipe_error = exc
                self.refresh_connection()
                continue
            except Exception as exc:
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
        self._logger.info(
            f"Query: {query} \n, parameters: {str(parameters)}\n, many: {many}"
        )

        with self.cursor() as cur:
            cur.executemany(query, parameters) if many else cur.execute(
                query, parameters
            )
            result = cur.fetchall()
            self._connection.commit()
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
        self._logger.info(
            f"Query: {query} \n, parameters: {str(parameters)}\n, many: {many}"
        )

        for _ in range(OCC_MAX_ATTEMPTS):
            with self.cursor() as cur:
                try:
                    cur.executemany(query, parameters) if many else cur.execute(
                        query, parameters
                    )
                    self._connection.commit()
                    break
                except pymonetdb.exceptions.IntegrityError as exc:
                    integrity_error = exc
                    self._connection.rollback()
                    sleep(INTEGRITY_ERROR_RETRY_INTERVAL)
                    continue
                except Exception as exc:
                    self._connection.rollback()
                    raise exc
        else:
            raise integrity_error

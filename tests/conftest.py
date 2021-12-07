import time

import pytest
import docker
import sqlalchemy as sql


TESTING_CONT_IMAGE = "madgik/mipenginedb:latest"
TESTING_CONT_NAME = "mipenginedb-testing"
TESTING_CONT_PORT = "50456"


class MonetDBSetupError(Exception):
    """Raised when the MonetDB container is unable to start."""


@pytest.fixture(scope="session")
def monetdb_container():
    client = docker.from_env()
    try:
        container = client.containers.get(TESTING_CONT_NAME)
    except docker.errors.NotFound:
        container = client.containers.run(
            TESTING_CONT_IMAGE,
            detach=True,
            ports={"50000/tcp": TESTING_CONT_PORT},
            name=TESTING_CONT_NAME,
            publish_all_ports=True,
        )
    # The time needed to start a monetdb container varies considerably. We need
    # to wait until some phrase appear in the logs to avoid starting the tests
    # too soon. The process is abandoned after 100 tries (50 sec).
    for _ in range(100):
        if b"new database mapi:monetdb" in container.logs():
            break
        time.sleep(0.5)
    else:
        raise MonetDBSetupError
    yield
    container = client.containers.get(TESTING_CONT_NAME)
    container.remove(v=True, force=True)


@pytest.fixture(scope="session")
def db():
    class MonetDBTesting:
        """MonetDB class used for testing."""

        def __init__(self) -> None:
            username = "monetdb"
            password = "monetdb"
            # ip = "172.17.0.1"
            port = TESTING_CONT_PORT
            dbfarm = "db"
            url = f"monetdb://{username}:{password}@localhost:{port}/{dbfarm}:"
            self._executor = sql.create_engine(url, echo=True)

        def execute(self, query, *args, **kwargs) -> list:
            return self._executor.execute(query, *args, **kwargs)

    return MonetDBTesting()


@pytest.fixture(scope="function")
def clean_db(db):
    yield
    select_user_tables = "SELECT name FROM sys.tables WHERE system=FALSE"
    user_tables = db.execute(select_user_tables).fetchall()
    for table_name, *_ in user_tables:
        db.execute(f"DROP TABLE {table_name}")


@pytest.fixture(scope="function")
def use_database(monetdb_container, clean_db):
    pass

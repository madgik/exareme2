from pathlib import Path

import pytest
import docker
import time
import subprocess
import os
from os import path
import toml
import sqlalchemy as sql

from mipengine.controller.node_tasks_handler_celery import NodeTasksHandlerCelery
import tests

ALGORITHM_FOLDERS_ENV_VARIABLE_VALUE = "./mipengine/algorithms,./tests/algorithms"
TESTING_RABBITMQ_CONT_IMAGE = "madgik/mipengine_rabbitmq:latest"
TESTING_MONETDB_CONT_IMAGE = "madgik/mipenginedb:latest"

this_mod_path = os.path.dirname(os.path.abspath(__file__))
TEST_ENV_CONFIG_FOLDER = path.join(this_mod_path, "testing_env_configs")
OUTDIR = Path("/tmp/mipengine/")
if not OUTDIR.exists():
    OUTDIR.mkdir()

DEMO_DATA_FOLDER = Path(tests.__file__).parent / "demo_data"

RABBITMQ_TMP_LOCALNODE_NAME = "rabbitmq_test_tmp_localnode"
RABBITMQ_TMP_LOCALNODE_PORT = 60003
MONETDB_TMP_LOCALNODE_NAME = "monetdb_test_tmp_localnode"
MONETDB_TMP_LOCALNODE_PORT = 61003
LOCALNODE_TMP_CONFIG_FILE = "tmp_localnode.toml"

TASKS_TIMEOUT = 60


class MonetDBSetupError(Exception):
    """Raised when the MonetDB container is unable to start."""


def _create_monetdb_container(cont_name, cont_port):
    client = docker.from_env()
    try:
        container = client.containers.get(cont_name)
    except docker.errors.NotFound:
        container = client.containers.run(
            TESTING_MONETDB_CONT_IMAGE,
            detach=True,
            ports={"50000/tcp": cont_port},
            name=cont_name,
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
        raise MonetDBSetupError()


def _remove_monetdb_container(cont_name):
    client = docker.from_env()
    container = client.containers.get(cont_name)
    container.remove(v=True, force=True)


@pytest.fixture(scope="function")
def monetdb_tmp_localnode():
    cont_name = MONETDB_TMP_LOCALNODE_NAME
    cont_port = MONETDB_TMP_LOCALNODE_PORT
    _create_monetdb_container(cont_name, cont_port)
    yield
    _remove_monetdb_container(cont_name)


@pytest.fixture(scope="function")
def monetdb_tmp_localnode_load_data(monetdb_tmp_localnode):

    config_file = Path(TEST_ENV_CONFIG_FOLDER) / LOCALNODE_TMP_CONFIG_FILE
    with open(config_file) as fp:
        tmp = toml.load(fp)
        db_ip = tmp["monetdb"]["ip"]
        db_port = tmp["monetdb"]["port"]

        # init the db
        cmd = f"mipdb init --ip {db_ip} --port {db_port} "
        subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)

        # load the data
        cmd = f"mipdb load-folder {DEMO_DATA_FOLDER}  --ip {db_ip} --port {db_port} "
        subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)

        import time

        time.sleep(60)


def _create_db_cursor(db_port):
    class MonetDBTesting:
        """MonetDB class used for testing."""

        def __init__(self) -> None:
            username = "monetdb"
            password = "monetdb"
            # ip = "172.17.0.1"
            port = db_port
            dbfarm = "db"
            url = f"monetdb://{username}:{password}@localhost:{port}/{dbfarm}:"
            self._executor = sql.create_engine(url, echo=True)

        def execute(self, query, *args, **kwargs) -> list:
            return self._executor.execute(query, *args, **kwargs)

    return MonetDBTesting()


@pytest.fixture(scope="function")
def localnode_tmp_db_cursor():
    return _create_db_cursor(MONETDB_TMP_LOCALNODE_PORT)


def _clean_db(cursor):
    select_user_tables = "SELECT name FROM sys.tables WHERE system=FALSE"
    user_tables = cursor.execute(select_user_tables).fetchall()
    for table_name, *_ in user_tables:
        cursor.execute(f"DROP TABLE {table_name} CASCADE")


def _create_rabbitmq_container(cont_name, cont_port):
    client = docker.from_env()
    try:
        container = client.containers.get(cont_name)
    except docker.errors.NotFound:
        container = client.containers.run(
            TESTING_RABBITMQ_CONT_IMAGE,
            detach=True,
            ports={"5672/tcp": cont_port},
            name=cont_name,
        )

    while (
        "Health" not in container.attrs["State"]
        or container.attrs["State"]["Health"]["Status"] != "healthy"
    ):
        container.reload()  # attributes are cached, this refreshes them..
        time.sleep(1)


def _remove_rabbitmq_container(cont_name):
    try:
        client = docker.from_env()
        container = client.containers.get(cont_name)
        container.remove(v=True, force=True)
    except docker.errors.NotFound:
        pass  # container was removed by other means..


@pytest.fixture(scope="function")
def rabbitmq_tmp_localnode():
    cont_name = RABBITMQ_TMP_LOCALNODE_NAME
    cont_port = RABBITMQ_TMP_LOCALNODE_PORT
    _create_rabbitmq_container(cont_name, cont_port)
    yield
    _remove_rabbitmq_container(cont_name)


def remove_tmp_localnode_rabbitmq():
    cont_name = RABBITMQ_TMP_LOCALNODE_NAME
    _remove_rabbitmq_container(cont_name)


def create_testtmplocalnode_node_service():  # algo_folders_env_variable_val, node_config_filepath):

    node_config_file = LOCALNODE_TMP_CONFIG_FILE
    algo_folders_env_variable_val = ALGORITHM_FOLDERS_ENV_VARIABLE_VALUE
    node_config_filepath = path.join(TEST_ENV_CONFIG_FOLDER, node_config_file)

    with open(node_config_filepath) as fp:
        tmp = toml.load(fp)
        node_id = tmp["identifier"]
    logpath = OUTDIR / (node_id + ".out")

    env = os.environ.copy()
    env["ALGORITHM_FOLDERS"] = algo_folders_env_variable_val
    env["MIPENGINE_NODE_CONFIG_FILE"] = node_config_filepath

    cmd = f"poetry run celery -A mipengine.node.node worker -l DEBUG >> {logpath} --purge 2>&1 "

    # if executed without "exec" it is spawned as a child process of the shell and it is
    # difficult to kill it
    # https://stackoverflow.com/questions/4789837/how-to-terminate-a-python-subprocess-launched-with-shell-true
    proc = subprocess.Popen(
        "exec " + cmd,
        shell=True,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        env=env,
    )

    # the celery app needs sometime to be ready, we should have some kind of check
    # for that, for now just a sleep..
    time.sleep(10)

    return proc


def kill_node_service(proc):
    proc.kill()


def get_testtmplocalnode_tasks_handler() -> NodeTasksHandlerCelery:  # tmp_localnode_node_service):
    node_config_filepath = path.join(TEST_ENV_CONFIG_FOLDER, LOCALNODE_TMP_CONFIG_FILE)

    with open(node_config_filepath) as fp:
        tmp = toml.load(fp)
        node_id = tmp["identifier"]
        queue_domain = tmp["rabbitmq"]["ip"]
        queue_port = tmp["rabbitmq"]["port"]
        db_domain = tmp["monetdb"]["ip"]
        db_port = tmp["monetdb"]["port"]
    queue_address = ":".join([str(queue_domain), str(queue_port)])
    db_address = ":".join([str(db_domain), str(db_port)])

    return NodeTasksHandlerCelery(
        node_id=node_id,
        node_queue_addr=queue_address,
        node_db_addr=db_address,
        tasks_timeout=TASKS_TIMEOUT,
    )


# @pytest.fixture(scope="function")
# def tmp_localnode_node_service(rabbitmq_tmp_localnode, monetdb_tmp_localnode):
#     """
#     ATTENTION!
#     This node service fixture is the only one returning the process so it can be killed.
#     The scope of the fixture is function so it won't break tests if the node service is killed.
#     The rabbitmq and monetdb containers have also function scope so this is VERY slow.
#     This should be used only when the service should be killed etc for testing.
#     """
#     # node_config_file = LOCALNODE_TMP_CONFIG_FILE
#     # algo_folders_env_variable_val = ALGORITHM_FOLDERS_ENV_VARIABLE_VALUE
#     # node_config_filepath = path.join(TEST_ENV_CONFIG_FOLDER, node_config_file)
#     # proc = _create_node_service(algo_folders_env_variable_val, node_config_filepath)
#     proc=_create_node_service()
#     yield proc
#     kill_node_service(proc)


# @pytest.fixture(scope="function")
# def tmp_localnode_tasks_handler(tmp_localnode_node_service):
#     node_config_filepath = path.join(TEST_ENV_CONFIG_FOLDER, LOCALNODE_TMP_CONFIG_FILE)

#     with open(node_config_filepath) as fp:
#         tmp = toml.load(fp)
#         node_id = tmp["identifier"]
#         queue_domain = tmp["rabbitmq"]["ip"]
#         queue_port = tmp["rabbitmq"]["port"]
#         db_domain = tmp["monetdb"]["ip"]
#         db_port = tmp["monetdb"]["port"]
#     queue_address = ":".join([str(queue_domain), str(queue_port)])
#     db_address = ":".join([str(db_domain), str(db_port)])

#     return NodeTasksHandlerCelery(
#         node_id=node_id,
#         node_queue_addr=queue_address,
#         node_db_addr=db_address,
#         tasks_timeout=TASKS_TIMEOUT,
#     )


# ---------------------------------------------------------------------------------
# import time

# import pytest
# import docker
# import sqlalchemy as sql


# TESTING_CONT_IMAGE = "madgik/mipenginedb:latest"
# TESTING_CONT_NAME = "mipenginedb-testing"
# TESTING_CONT_PORT = "50456"


# class MonetDBSetupError(Exception):
#     """Raised when the MonetDB container is unable to start."""


# @pytest.fixture(scope="session")
# def monetdb_container():
#     client = docker.from_env()
#     try:
#         container = client.containers.get(TESTING_CONT_NAME)
#     except docker.errors.NotFound:
#         container = client.containers.run(
#             TESTING_CONT_IMAGE,
#             detach=True,
#             ports={"50000/tcp": TESTING_CONT_PORT},
#             name=TESTING_CONT_NAME,
#             publish_all_ports=True,
#         )
#     # The time needed to start a monetdb container varies considerably. We need
#     # to wait until some phrase appear in the logs to avoid starting the tests
#     # too soon. The process is abandoned after 100 tries (50 sec).
#     for _ in range(100):
#         if b"new database mapi:monetdb" in container.logs():
#             break
#         time.sleep(0.5)
#     else:
#         raise MonetDBSetupError
#     yield
#     container = client.containers.get(TESTING_CONT_NAME)
#     container.remove(v=True, force=True)


# @pytest.fixture(scope="session")
# def db():
#     class MonetDBTesting:
#         """MonetDB class used for testing."""

#         def __init__(self) -> None:
#             username = "monetdb"
#             password = "monetdb"
#             # ip = "172.17.0.1"
#             port = TESTING_CONT_PORT
#             dbfarm = "db"
#             url = f"monetdb://{username}:{password}@localhost:{port}/{dbfarm}:"
#             self._executor = sql.create_engine(url, echo=True)

#         def execute(self, query, *args, **kwargs) -> list:
#             return self._executor.execute(query, *args, **kwargs)

#     return MonetDBTesting()


# @pytest.fixture(scope="function")
# def clean_db(db):
#     yield
#     select_user_tables = "SELECT name FROM sys.tables WHERE system=FALSE"
#     user_tables = db.execute(select_user_tables).fetchall()
#     for table_name, *_ in user_tables:
#         db.execute(f"DROP TABLE {table_name}")


# @pytest.fixture(scope="function")
# def use_database(monetdb_container, clean_db):
#     pass

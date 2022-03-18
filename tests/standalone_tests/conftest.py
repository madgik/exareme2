import os
import subprocess
import time
from os import path
from pathlib import Path

import docker
import pytest
import sqlalchemy as sql
import toml

from mipengine.controller.node_tasks_handler_celery import NodeTasksHandlerCelery
from mipengine.udfgen import udfio

ALGORITHM_FOLDERS_ENV_VARIABLE_VALUE = "./mipengine/algorithms,./tests/algorithms"
TESTING_RABBITMQ_CONT_IMAGE = "madgik/mipengine_rabbitmq:latest"
TESTING_MONETDB_CONT_IMAGE = "madgik/mipenginedb:latest"

this_mod_path = os.path.dirname(os.path.abspath(__file__))
TEST_ENV_CONFIG_FOLDER = path.join(this_mod_path, "testing_env_configs")
TEST_DATA_FOLDER = Path(this_mod_path).parent / "demo_data"

OUTDIR = Path("/tmp/mipengine/")
if not OUTDIR.exists():
    OUTDIR.mkdir()

COMMON_IP = "172.17.0.1"
RABBITMQ_GLOBALNODE_NAME = "rabbitmq_test_globalnode"
RABBITMQ_LOCALNODE1_NAME = "rabbitmq_test_localnode1"
RABBITMQ_LOCALNODE2_NAME = "rabbitmq_test_localnode2"

RABBITMQ_LOCALNODETMP_NAME = "rabbitmq_test_localnodetmp"
RABBITMQ_SMPC_GLOBALNODE_NAME = "rabbitmq_test_smpc_globalnode"
RABBITMQ_SMPC_LOCALNODE1_NAME = "rabbitmq_test_smpc_localnode1"
RABBITMQ_SMPC_LOCALNODE2_NAME = "rabbitmq_test_smpc_localnode2"

RABBITMQ_GLOBALNODE_PORT = 60000
RABBITMQ_LOCALNODE1_PORT = 60001
RABBITMQ_LOCALNODE2_PORT = 60002
RABBITMQ_LOCALNODETMP_PORT = 60003
RABBITMQ_SMPC_GLOBALNODE_PORT = 60004
RABBITMQ_SMPC_LOCALNODE1_PORT = 60005
RABBITMQ_SMPC_LOCALNODE2_PORT = 60006

MONETDB_GLOBALNODE_NAME = "monetdb_test_globalnode"
MONETDB_LOCALNODE1_NAME = "monetdb_test_localnode1"
MONETDB_LOCALNODE2_NAME = "monetdb_test_localnode2"
MONETDB_LOCALNODETMP_NAME = "monetdb_test_localnodetmp"
MONETDB_SMPC_GLOBALNODE_NAME = "monetdb_test_smpc_globalnode"
MONETDB_SMPC_LOCALNODE1_NAME = "monetdb_test_smpc_localnode1"
MONETDB_SMPC_LOCALNODE2_NAME = "monetdb_test_smpc_localnode2"
MONETDB_GLOBALNODE_PORT = 61000
MONETDB_LOCALNODE1_PORT = 61001
MONETDB_LOCALNODE2_PORT = 61002
MONETDB_LOCALNODETMP_PORT = 61003
MONETDB_SMPC_GLOBALNODE_PORT = 61004
MONETDB_SMPC_LOCALNODE1_PORT = 61005
MONETDB_SMPC_LOCALNODE2_PORT = 61006

GLOBALNODE_CONFIG_FILE = "testglobalnode.toml"
LOCALNODE1_CONFIG_FILE = "testlocalnode1.toml"
LOCALNODE2_CONFIG_FILE = "testlocalnode2.toml"
LOCALNODETMP_CONFIG_FILE = "testlocalnodetmp.toml"
GLOBALNODE_SMPC_CONFIG_FILE = "smpc_globalnode.toml"
LOCALNODE1_SMPC_CONFIG_FILE = "smpc_localnode1.toml"
LOCALNODE2_SMPC_CONFIG_FILE = "smpc_localnode2.toml"


# TODO Instead of the fixtures having scope session, it could be function,
# but when the fixture start, it should check if it already exists, thus
# not creating it again (fast). This could solve the problem of some
# tests destroying some containers to test things.


class MonetDBSetupError(Exception):
    """Raised when the MonetDB container is unable to start."""


def _create_monetdb_container(cont_name, cont_port):
    client = docker.from_env()
    try:
        container = client.containers.get(cont_name)
    except docker.errors.NotFound:
        udfio_full_path = path.abspath(udfio.__file__)
        # A volume is used to pass the udfio inside the monetdb container.
        # This is done so that we don't need to rebuild every time the udfio.py file is changed.
        container = client.containers.run(
            TESTING_MONETDB_CONT_IMAGE,
            detach=True,
            ports={"50000/tcp": cont_port},
            volumes=[f"{udfio_full_path}:/home/udflib/udfio.py"],
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
        raise MonetDBSetupError


def _remove_monetdb_container(cont_name):
    client = docker.from_env()
    container = client.containers.get(cont_name)
    container.remove(v=True, force=True)


@pytest.fixture(scope="session")
def monetdb_globalnode():
    cont_name = MONETDB_GLOBALNODE_NAME
    cont_port = MONETDB_GLOBALNODE_PORT
    _create_monetdb_container(cont_name, cont_port)
    yield
    # TODO Very slow development if containers are always removed afterwards
    # _remove_monetdb_container(cont_name)


@pytest.fixture(scope="session")
def monetdb_localnode1():
    cont_name = MONETDB_LOCALNODE1_NAME
    cont_port = MONETDB_LOCALNODE1_PORT
    _create_monetdb_container(cont_name, cont_port)
    yield
    # TODO Very slow development if containers are always removed afterwards
    # _remove_monetdb_container(cont_name)


@pytest.fixture(scope="session")
def monetdb_localnode2():
    cont_name = MONETDB_LOCALNODE2_NAME
    cont_port = MONETDB_LOCALNODE2_PORT
    _create_monetdb_container(cont_name, cont_port)
    yield
    # TODO Very slow development if containers are always removed afterwards
    # _remove_monetdb_container(cont_name)


@pytest.fixture(scope="session")
def monetdb_smpc_globalnode():
    cont_name = MONETDB_SMPC_GLOBALNODE_NAME
    cont_port = MONETDB_SMPC_GLOBALNODE_PORT
    _create_monetdb_container(cont_name, cont_port)
    yield
    # TODO Very slow development if containers are always removed afterwards
    # _remove_monetdb_container(cont_name)


@pytest.fixture(scope="session")
def monetdb_smpc_localnode1():
    cont_name = MONETDB_SMPC_LOCALNODE1_NAME
    cont_port = MONETDB_SMPC_LOCALNODE1_PORT
    _create_monetdb_container(cont_name, cont_port)
    yield
    # TODO Very slow development if containers are always removed afterwards
    # _remove_monetdb_container(cont_name)


@pytest.fixture(scope="session")
def monetdb_smpc_localnode2():
    cont_name = MONETDB_SMPC_LOCALNODE2_NAME
    cont_port = MONETDB_SMPC_LOCALNODE2_PORT
    _create_monetdb_container(cont_name, cont_port)
    yield
    # TODO Very slow development if containers are always removed afterwards
    # _remove_monetdb_container(cont_name)


@pytest.fixture(scope="function")
def monetdb_localnodetmp():
    cont_name = MONETDB_LOCALNODETMP_NAME
    cont_port = MONETDB_LOCALNODETMP_PORT
    _create_monetdb_container(cont_name, cont_port)
    yield
    _remove_monetdb_container(cont_name)


def _init_database_monetdb_container(db_ip, db_port):
    # init the db
    cmd = f"mipdb init --ip {db_ip} --port {db_port} "
    subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)


def _load_data_monetdb_container(db_ip, db_port):
    # load the data
    cmd = f"mipdb load-folder {TEST_DATA_FOLDER}  --ip {db_ip} --port {db_port} "
    subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)


def _remove_data_model_from_monetdb_container(
    data_model_code, data_model_version, db_ip, db_port
):
    # Remove data_model
    cmd = f"mipdb delete-data-model {data_model_code} -v {data_model_version} -f  --ip {db_ip} --port {db_port} "
    subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)


@pytest.fixture(scope="session")
def init_data_globalnode(monetdb_globalnode):
    _init_database_monetdb_container(COMMON_IP, MONETDB_GLOBALNODE_PORT)
    yield


@pytest.fixture(scope="session")
def load_data_localnode1(monetdb_localnode1):
    _init_database_monetdb_container(COMMON_IP, MONETDB_LOCALNODE1_PORT)
    _load_data_monetdb_container(COMMON_IP, MONETDB_LOCALNODE1_PORT)
    yield


@pytest.fixture(scope="session")
def load_data_localnode2(monetdb_localnode2):
    _init_database_monetdb_container(COMMON_IP, MONETDB_LOCALNODE2_PORT)
    _load_data_monetdb_container(COMMON_IP, MONETDB_LOCALNODE2_PORT)
    yield


@pytest.fixture(scope="function")
def load_data_localnodetmp(monetdb_localnodetmp):
    _init_database_monetdb_container(COMMON_IP, MONETDB_LOCALNODETMP_PORT)
    _load_data_monetdb_container(COMMON_IP, MONETDB_LOCALNODETMP_PORT)
    yield


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


@pytest.fixture(scope="session")
def globalnode_db_cursor():
    return _create_db_cursor(MONETDB_GLOBALNODE_PORT)


@pytest.fixture(scope="session")
def localnode1_db_cursor():
    return _create_db_cursor(MONETDB_LOCALNODE1_PORT)


@pytest.fixture(scope="session")
def localnode2_db_cursor():
    return _create_db_cursor(MONETDB_LOCALNODE2_PORT)


@pytest.fixture(scope="session")
def globalnode_smpc_db_cursor():
    return _create_db_cursor(MONETDB_SMPC_GLOBALNODE_PORT)


@pytest.fixture(scope="session")
def localnode1_smpc_db_cursor():
    return _create_db_cursor(MONETDB_SMPC_LOCALNODE1_PORT)


@pytest.fixture(scope="session")
def localnode2_smpc_db_cursor():
    return _create_db_cursor(MONETDB_SMPC_LOCALNODE2_PORT)


@pytest.fixture(scope="function")
def localnodetmp_db_cursor():
    return _create_db_cursor(MONETDB_LOCALNODETMP_PORT)


def _clean_db(cursor):
    # schema_id=2000 is the default schema id
    select_user_tables = (
        "SELECT name FROM sys.tables WHERE system=FALSE AND schema_id=2000"
    )
    user_tables = cursor.execute(select_user_tables).fetchall()
    for table_name, *_ in user_tables:
        cursor.execute(f"DROP TABLE {table_name} CASCADE")


@pytest.fixture(scope="function")
def clean_globalnode_db(globalnode_db_cursor):
    yield
    _clean_db(globalnode_db_cursor)


@pytest.fixture(scope="function")
def clean_localnode1_db(localnode1_db_cursor):
    yield
    _clean_db(localnode1_db_cursor)


@pytest.fixture(scope="function")
def clean_smpc_globalnode_db(globalnode_smpc_db_cursor):
    yield
    _clean_db(globalnode_smpc_db_cursor)


@pytest.fixture(scope="function")
def clean_smpc_localnode1_db(localnode1_smpc_db_cursor):
    yield
    _clean_db(localnode1_smpc_db_cursor)


@pytest.fixture(scope="function")
def clean_smpc_localnode2_db(localnode2_smpc_db_cursor):
    yield
    _clean_db(localnode2_smpc_db_cursor)


@pytest.fixture(scope="function")
def clean_localnode2_db(localnode2_db_cursor):
    yield
    _clean_db(localnode2_db_cursor)


@pytest.fixture(scope="function")
def use_globalnode_database(monetdb_globalnode, clean_globalnode_db):
    pass


@pytest.fixture(scope="function")
def use_localnode1_database(monetdb_localnode1, clean_localnode1_db):
    pass


@pytest.fixture(scope="function")
def use_localnode2_database(monetdb_localnode2, clean_localnode2_db):
    pass


@pytest.fixture(scope="function")
def use_smpc_globalnode_database(monetdb_smpc_globalnode, clean_smpc_globalnode_db):
    pass


@pytest.fixture(scope="function")
def use_smpc_localnode1_database(monetdb_smpc_localnode1, clean_smpc_localnode1_db):
    pass


@pytest.fixture(scope="function")
def use_smpc_localnode2_database(monetdb_smpc_localnode2, clean_smpc_localnode2_db):
    pass


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


@pytest.fixture(scope="session")
def rabbitmq_globalnode():
    cont_name = RABBITMQ_GLOBALNODE_NAME
    cont_port = RABBITMQ_GLOBALNODE_PORT
    _create_rabbitmq_container(cont_name, cont_port)
    yield
    # TODO Very slow development if containers are always removed afterwards
    # _remove_rabbitmq_container(cont_name)


@pytest.fixture(scope="session")
def rabbitmq_localnode1():
    cont_name = RABBITMQ_LOCALNODE1_NAME
    cont_port = RABBITMQ_LOCALNODE1_PORT
    _create_rabbitmq_container(cont_name, cont_port)
    yield
    # TODO Very slow development if containers are always removed afterwards
    # _remove_rabbitmq_container(cont_name)


@pytest.fixture(scope="session")
def rabbitmq_localnode2():
    cont_name = RABBITMQ_LOCALNODE2_NAME
    cont_port = RABBITMQ_LOCALNODE2_PORT
    _create_rabbitmq_container(cont_name, cont_port)
    yield
    # TODO Very slow development if containers are always removed afterwards
    # _remove_rabbitmq_container(cont_name)


@pytest.fixture(scope="session")
def rabbitmq_smpc_globalnode():
    cont_name = RABBITMQ_SMPC_GLOBALNODE_NAME
    cont_port = RABBITMQ_SMPC_GLOBALNODE_PORT
    _create_rabbitmq_container(cont_name, cont_port)
    yield
    # TODO Very slow development if containers are always removed afterwards
    # _remove_rabbitmq_container(cont_name)


@pytest.fixture(scope="session")
def rabbitmq_smpc_localnode1():
    cont_name = RABBITMQ_SMPC_LOCALNODE1_NAME
    cont_port = RABBITMQ_SMPC_LOCALNODE1_PORT
    _create_rabbitmq_container(cont_name, cont_port)
    yield
    # TODO Very slow development if containers are always removed afterwards
    # _remove_rabbitmq_container(cont_name)


@pytest.fixture(scope="session")
def rabbitmq_smpc_localnode2():
    cont_name = RABBITMQ_SMPC_LOCALNODE2_NAME
    cont_port = RABBITMQ_SMPC_LOCALNODE2_PORT
    _create_rabbitmq_container(cont_name, cont_port)
    yield
    # TODO Very slow development if containers are always removed afterwards
    # _remove_rabbitmq_container(cont_name)


@pytest.fixture(scope="function")
def rabbitmq_localnodetmp():
    cont_name = RABBITMQ_LOCALNODETMP_NAME
    cont_port = RABBITMQ_LOCALNODETMP_PORT
    _create_rabbitmq_container(cont_name, cont_port)
    yield
    _remove_rabbitmq_container(cont_name)


def remove_localnodetmp_rabbitmq():
    cont_name = RABBITMQ_LOCALNODETMP_NAME
    _remove_rabbitmq_container(cont_name)


def _create_node_service(algo_folders_env_variable_val, node_config_filepath):
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
    # might take some time for the celery service to be killed
    time.sleep(10)


@pytest.fixture(scope="session")
def globalnode_node_service(rabbitmq_globalnode, monetdb_globalnode):
    node_config_file = GLOBALNODE_CONFIG_FILE
    algo_folders_env_variable_val = ALGORITHM_FOLDERS_ENV_VARIABLE_VALUE
    node_config_filepath = path.join(TEST_ENV_CONFIG_FOLDER, node_config_file)
    proc = _create_node_service(algo_folders_env_variable_val, node_config_filepath)
    yield
    kill_node_service(proc)


@pytest.fixture(scope="session")
def localnode1_node_service(rabbitmq_localnode1, monetdb_localnode1):
    node_config_file = LOCALNODE1_CONFIG_FILE
    algo_folders_env_variable_val = ALGORITHM_FOLDERS_ENV_VARIABLE_VALUE
    node_config_filepath = path.join(TEST_ENV_CONFIG_FOLDER, node_config_file)
    proc = _create_node_service(algo_folders_env_variable_val, node_config_filepath)
    yield
    kill_node_service(proc)


@pytest.fixture(scope="session")
def localnode2_node_service(rabbitmq_localnode2, monetdb_localnode2):
    node_config_file = LOCALNODE2_CONFIG_FILE
    algo_folders_env_variable_val = ALGORITHM_FOLDERS_ENV_VARIABLE_VALUE
    node_config_filepath = path.join(TEST_ENV_CONFIG_FOLDER, node_config_file)
    proc = _create_node_service(algo_folders_env_variable_val, node_config_filepath)
    yield
    kill_node_service(proc)


@pytest.fixture(scope="session")
def smpc_globalnode_node_service(rabbitmq_smpc_globalnode, monetdb_smpc_globalnode):
    node_config_file = GLOBALNODE_SMPC_CONFIG_FILE
    algo_folders_env_variable_val = ALGORITHM_FOLDERS_ENV_VARIABLE_VALUE
    node_config_filepath = path.join(TEST_ENV_CONFIG_FOLDER, node_config_file)
    proc = _create_node_service(algo_folders_env_variable_val, node_config_filepath)
    yield
    kill_node_service(proc)


@pytest.fixture(scope="session")
def smpc_localnode1_node_service(rabbitmq_smpc_localnode1, monetdb_smpc_localnode1):
    node_config_file = LOCALNODE1_SMPC_CONFIG_FILE
    algo_folders_env_variable_val = ALGORITHM_FOLDERS_ENV_VARIABLE_VALUE
    node_config_filepath = path.join(TEST_ENV_CONFIG_FOLDER, node_config_file)
    proc = _create_node_service(algo_folders_env_variable_val, node_config_filepath)
    yield
    kill_node_service(proc)


@pytest.fixture(scope="session")
def smpc_localnode2_node_service(rabbitmq_smpc_localnode2, monetdb_smpc_localnode2):
    node_config_file = LOCALNODE2_SMPC_CONFIG_FILE
    algo_folders_env_variable_val = ALGORITHM_FOLDERS_ENV_VARIABLE_VALUE
    node_config_filepath = path.join(TEST_ENV_CONFIG_FOLDER, node_config_file)
    proc = _create_node_service(algo_folders_env_variable_val, node_config_filepath)
    yield
    kill_node_service(proc)


@pytest.fixture(scope="function")
def localnodetmp_node_service(rabbitmq_localnodetmp, monetdb_localnodetmp):
    """
    ATTENTION!
    This node service fixture is the only one returning the process so it can be killed.
    The scope of the fixture is function so it won't break tests if the node service is killed.
    The rabbitmq and monetdb containers have also function scope so this is VERY slow.
    This should be used only when the service should be killed etc for testing.
    """
    node_config_file = LOCALNODETMP_CONFIG_FILE
    algo_folders_env_variable_val = ALGORITHM_FOLDERS_ENV_VARIABLE_VALUE
    node_config_filepath = path.join(TEST_ENV_CONFIG_FOLDER, node_config_file)
    proc = _create_node_service(algo_folders_env_variable_val, node_config_filepath)
    yield proc
    kill_node_service(proc)


@pytest.fixture(scope="session")
def globalnode_tasks_handler_celery(globalnode_node_service):
    node_config_filepath = path.join(TEST_ENV_CONFIG_FOLDER, GLOBALNODE_CONFIG_FILE)

    with open(node_config_filepath) as fp:
        tmp = toml.load(fp)
        node_id = tmp["identifier"]
        queue_domain = tmp["rabbitmq"]["ip"]
        queue_port = tmp["rabbitmq"]["port"]
        db_domain = tmp["monetdb"]["ip"]
        db_port = tmp["monetdb"]["port"]
        tasks_timeout = tmp["celery"]["task_time_limit"]
    queue_address = ":".join([str(queue_domain), str(queue_port)])
    db_address = ":".join([str(db_domain), str(db_port)])

    return NodeTasksHandlerCelery(
        node_id=node_id,
        node_queue_addr=queue_address,
        node_db_addr=db_address,
        tasks_timeout=tasks_timeout,
    )


@pytest.fixture(scope="session")
def localnode1_tasks_handler_celery(localnode1_node_service):
    node_config_filepath = path.join(TEST_ENV_CONFIG_FOLDER, LOCALNODE1_CONFIG_FILE)
    with open(node_config_filepath) as fp:
        tmp = toml.load(fp)
        node_id = tmp["identifier"]
        queue_domain = tmp["rabbitmq"]["ip"]
        queue_port = tmp["rabbitmq"]["port"]
        db_domain = tmp["monetdb"]["ip"]
        db_port = tmp["monetdb"]["port"]
        tasks_timeout = tmp["celery"]["task_time_limit"]

    queue_address = ":".join([str(queue_domain), str(queue_port)])
    db_address = ":".join([str(db_domain), str(db_port)])

    return NodeTasksHandlerCelery(
        node_id=node_id,
        node_queue_addr=queue_address,
        node_db_addr=db_address,
        tasks_timeout=tasks_timeout,
    )


@pytest.fixture(scope="function")
def localnodetmp_tasks_handler_celery(localnodetmp_node_service):
    node_config_filepath = path.join(TEST_ENV_CONFIG_FOLDER, LOCALNODETMP_CONFIG_FILE)
    with open(node_config_filepath) as fp:
        tmp = toml.load(fp)
        node_id = tmp["identifier"]
        queue_domain = tmp["rabbitmq"]["ip"]
        queue_port = tmp["rabbitmq"]["port"]
        db_domain = tmp["monetdb"]["ip"]
        db_port = tmp["monetdb"]["port"]
        tasks_timeout = tmp["celery"]["task_time_limit"]
    queue_address = ":".join([str(queue_domain), str(queue_port)])
    db_address = ":".join([str(db_domain), str(db_port)])

    return NodeTasksHandlerCelery(
        node_id=node_id,
        node_queue_addr=queue_address,
        node_db_addr=db_address,
        tasks_timeout=tasks_timeout,
    )

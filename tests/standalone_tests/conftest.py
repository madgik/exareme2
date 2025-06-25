import enum
import json
import os
import pathlib
import re
import sqlite3
import subprocess
import time
from itertools import chain
from os import path
from pathlib import Path
from typing import List
from typing import Union

import docker
import psutil
import pytest
import sqlalchemy as sql
import toml

from exareme2 import AttrDict
from exareme2.algorithms.exareme2.udfgen import udfio
from exareme2.controller.celery.app import CeleryAppFactory
from exareme2.controller.logger import init_logger
from exareme2.controller.services.exareme2.tasks_handler import Exareme2TasksHandler
from exareme2.worker_communication import TableSchema

EXAREME2_ALGORITHM_FOLDERS_ENV_VARIABLE_VALUE = (
    "./exareme2/algorithms/exareme2,./tests/algorithms/exareme2"
)
FLOWER_ALGORITHM_FOLDERS_ENV_VARIABLE_VALUE = (
    "./exareme2/algorithms/flower,./tests/algorithms/flower"
)
EXAFLOW_ALGORITHM_FOLDERS_ENV_VARIABLE_VALUE = (
    "./exareme2/algorithms/exaflow,./tests/algorithms/exaflow"
)
TESTING_RABBITMQ_CONT_IMAGE = "madgik/exareme2_rabbitmq:dev"
TESTING_MONETDB_CONT_IMAGE = "madgik/exareme2_db:dev"

# This is used in the github actions CI. In CI images are built and not pulled.
PULL_DOCKER_IMAGES_STR = os.getenv("PULL_DOCKER_IMAGES", "true")
PULL_DOCKER_IMAGES = PULL_DOCKER_IMAGES_STR.lower() == "true"


this_mod_path = os.path.dirname(os.path.abspath(__file__))
TEST_ENV_CONFIG_FOLDER = path.join(this_mod_path, "testing_env_configs")
TEST_DATA_FOLDER = Path(this_mod_path).parent / "test_data"

OUTDIR = Path("/tmp/exareme2/")
if not OUTDIR.exists():
    OUTDIR.mkdir()

USE_EXTERNAL_SMPC_CLUSTER = True

COMMON_IP = "172.17.0.1"
COMMON_MONETDB_NAME = "db"
COMMON_MONETDB_USERNAME = "admin"
COMMON_MONETDB_PASSWORD = "executor"

ALGORITHMS_URL = f"http://{COMMON_IP}:4500/algorithms"
SMPC_ALGORITHMS_URL = f"http://{COMMON_IP}:4501/algorithms"

HEALTHCHECK_URL = f"http://{COMMON_IP}:4500/healthcheck"

RABBITMQ_GLOBALWORKER_NAME = "rabbitmq_test_globalworker"
RABBITMQ_LOCALWORKER1_NAME = "rabbitmq_test_localworker1"
RABBITMQ_LOCALWORKER2_NAME = "rabbitmq_test_localworker2"

RABBITMQ_LOCALWORKERTMP_NAME = "rabbitmq_test_localworkertmp"
RABBITMQ_SMPC_GLOBALWORKER_NAME = "rabbitmq_test_smpc_globalworker"
RABBITMQ_SMPC_LOCALWORKER1_NAME = "rabbitmq_test_smpc_localworker1"
RABBITMQ_SMPC_LOCALWORKER2_NAME = "rabbitmq_test_smpc_localworker2"

RABBITMQ_GLOBALWORKER_PORT = 60000
RABBITMQ_GLOBALWORKER_ADDR = f"{COMMON_IP}:{str(RABBITMQ_GLOBALWORKER_PORT)}"
RABBITMQ_LOCALWORKER1_PORT = 60001
RABBITMQ_LOCALWORKER1_ADDR = f"{COMMON_IP}:{str(RABBITMQ_LOCALWORKER1_PORT)}"
RABBITMQ_LOCALWORKER2_PORT = 60002
RABBITMQ_LOCALWORKER2_ADDR = f"{COMMON_IP}:{str(RABBITMQ_LOCALWORKER2_PORT)}"
RABBITMQ_LOCALWORKERTMP_PORT = 60003
RABBITMQ_LOCALWORKERTMP_ADDR = f"{COMMON_IP}:{str(RABBITMQ_LOCALWORKERTMP_PORT)}"
RABBITMQ_SMPC_GLOBALWORKER_PORT = 60004
RABBITMQ_SMPC_GLOBALWORKER_ADDR = f"{COMMON_IP}:{str(RABBITMQ_SMPC_GLOBALWORKER_PORT)}"
RABBITMQ_SMPC_LOCALWORKER1_PORT = 60005
RABBITMQ_SMPC_LOCALWORKER1_ADDR = f"{COMMON_IP}:{str(RABBITMQ_SMPC_LOCALWORKER1_PORT)}"
RABBITMQ_SMPC_LOCALWORKER2_PORT = 60006
RABBITMQ_SMPC_LOCALWORKER2_ADDR = f"{COMMON_IP}:{str(RABBITMQ_SMPC_LOCALWORKER2_PORT)}"

DATASET_SUFFIXES_LOCALWORKER1 = [0, 1, 2, 3]
DATASET_SUFFIXES_LOCALWORKER2 = [4, 5, 6]
DATASET_SUFFIXES_LOCALWORKERTMP = [7, 8, 9]
DATASET_SUFFIXES_SMPC_LOCALWORKER1 = [0, 1, 2, 3, 4]
DATASET_SUFFIXES_SMPC_LOCALWORKER2 = [5, 6, 7, 8, 9]
MONETDB_GLOBALWORKER_NAME = "monetdb_test_globalworker"
MONETDB_LOCALWORKER1_NAME = "monetdb_test_localworker1"
MONETDB_LOCALWORKER2_NAME = "monetdb_test_localworker2"
MONETDB_LOCALWORKERTMP_NAME = "monetdb_test_localworkertmp"
MONETDB_SMPC_GLOBALWORKER_NAME = "monetdb_test_smpc_globalworker"
MONETDB_SMPC_LOCALWORKER1_NAME = "monetdb_test_smpc_localworker1"
MONETDB_SMPC_LOCALWORKER2_NAME = "monetdb_test_smpc_localworker2"
MONETDB_GLOBALWORKER_PORT = 61000
MONETDB_GLOBALWORKER_ADDR = f"{COMMON_IP}:{str(MONETDB_GLOBALWORKER_PORT)}"
MONETDB_LOCALWORKER1_PORT = 61001
MONETDB_LOCALWORKER1_ADDR = f"{COMMON_IP}:{str(MONETDB_LOCALWORKER1_PORT)}"
MONETDB_LOCALWORKER2_PORT = 61002
MONETDB_LOCALWORKER2_ADDR = f"{COMMON_IP}:{str(MONETDB_LOCALWORKER2_PORT)}"
MONETDB_LOCALWORKERTMP_PORT = 61003
MONETDB_LOCALWORKERTMP_ADDR = f"{COMMON_IP}:{str(MONETDB_LOCALWORKERTMP_PORT)}"
MONETDB_SMPC_GLOBALWORKER_PORT = 61004
MONETDB_SMPC_LOCALWORKER1_PORT = 61005
MONETDB_SMPC_LOCALWORKER2_PORT = 61006
CONTROLLER_PORT = 4500
CONTROLLER_SMPC_PORT = 4501

GLOBALWORKER_CONFIG_FILE = "test_globalworker.toml"
LOCALWORKER1_CONFIG_FILE = "test_localworker1.toml"
LOCALWORKER2_CONFIG_FILE = "test_localworker2.toml"
LOCALWORKERTMP_CONFIG_FILE = "test_localworkertmp.toml"
CONTROLLER_CONFIG_FILE = "test_controller.toml"
AGG_SERVER_CONFIG_FILE = "test_aggregation_server.toml"
CONTROLLER_GLOBALWORKER_LOCALWORKER1_ADDRESSES_FILE = (
    "test_localworker1_globalworker_addresses.json"
)
CONTROLLER_LOCALWORKER1_ADDRESSES_FILE = "test_localworker1_addresses.json"
CONTROLLER_LOCALWORKERTMP_ADDRESSES_FILE = "test_localworkertmp_addresses.json"
CONTROLLER_GLOBALWORKER_LOCALWORKER1_LOCALWORKER2_ADDRESSES_FILE = (
    "test_globalworker_localworker1_localworker2_addresses.json"
)
CONTROLLER_GLOBALWORKER_LOCALWORKER1_LOCALWORKER2_LOCALWORKERTMP_ADDRESSES_FILE = (
    "test_globalworker_localworker1_localworker2_localworkertmp_addresses.json"
)
CONTROLLER_OUTPUT_FILE = "test_controller.out"
if USE_EXTERNAL_SMPC_CLUSTER:
    GLOBALWORKER_SMPC_CONFIG_FILE = "test_external_smpc_globalworker.toml"
    LOCALWORKER1_SMPC_CONFIG_FILE = "test_external_smpc_localworker1.toml"
    LOCALWORKER2_SMPC_CONFIG_FILE = "test_external_smpc_localworker2.toml"
    CONTROLLER_SMPC_CONFIG_FILE = "test_external_smpc_controller.toml"
    CONTROLLER_SMPC_DP_CONFIG_FILE = "test_external_smpc_dp_controller.toml"
    SMPC_COORDINATOR_ADDRESS = "http://167.71.139.232:12314"
else:
    GLOBALWORKER_SMPC_CONFIG_FILE = "test_smpc_globalworker.toml"
    LOCALWORKER1_SMPC_CONFIG_FILE = "test_smpc_localworker1.toml"
    LOCALWORKER2_SMPC_CONFIG_FILE = "test_smpc_localworker2.toml"
    CONTROLLER_SMPC_CONFIG_FILE = "test_smpc_controller.toml"
    SMPC_COORDINATOR_ADDRESS = "http://172.17.0.1:12314"
CONTROLLER_SMPC_LOCALWORKERS_CONFIG_FILE = "test_smpc_localworkers_addresses.json"
SMPC_CONTROLLER_OUTPUT_FILE = "test_smpc_controller.out"

TASKS_TIMEOUT = 10
RUN_UDF_TASK_TIMEOUT = 120
SMPC_CLUSTER_SLEEP_TIME = 60

REQUEST_ID = "STANDALONETEST"

# ------------ SMPC Cluster ------------ #

SMPC_CLUSTER_IMAGE = "gpikra/coordinator:v7.0.7.4"
SMPC_COORD_DB_IMAGE = "mongo:5.0.8"
SMPC_COORD_QUEUE_IMAGE = "redis:alpine3.15"

SMPC_COORD_CONT_NAME = "smpc_test_coordinator"
SMPC_COORD_DB_CONT_NAME = "smpc_test_coordinator_db"
SMPC_COORD_QUEUE_CONT_NAME = "smpc_test_coordinator_queue"
SMPC_PLAYER1_CONT_NAME = "smpc_test_player1"
SMPC_PLAYER2_CONT_NAME = "smpc_test_player2"
SMPC_PLAYER3_CONT_NAME = "smpc_test_player3"
SMPC_CLIENT1_CONT_NAME = "smpc_test_client1"
SMPC_CLIENT2_CONT_NAME = "smpc_test_client2"

SMPC_COORD_PORT = 12314
SMPC_COORD_DB_PORT = 27017
SMPC_COORD_QUEUE_PORT = 6379
SMPC_PLAYER1_PORT1 = 6000
SMPC_PLAYER1_PORT2 = 7000
SMPC_PLAYER1_PORT3 = 14000
SMPC_PLAYER2_PORT1 = 6001
SMPC_PLAYER2_PORT2 = 7001
SMPC_PLAYER2_PORT3 = 14001
SMPC_PLAYER3_PORT1 = 6002
SMPC_PLAYER3_PORT2 = 7002
SMPC_PLAYER3_PORT3 = 14002
SMPC_CLIENT1_PORT = 9005
SMPC_CLIENT2_PORT = 9006


#####################################


class MonetDBConfigurations:
    def __init__(self, port):
        self.ip = COMMON_IP
        self.port = port
        self.username = COMMON_MONETDB_USERNAME
        self.password = COMMON_MONETDB_PASSWORD
        self.database = COMMON_MONETDB_NAME

    def convert_to_mipdb_format(self):
        return (
            f"--ip {self.ip} "
            f"--port {self.port} "
            f"--username {self.username} "
            f"--password {self.password} "
            f"--db_name {self.database}"
        )


def _search_for_string_in_logfile(
    log_to_search_for: str, logspath: Path, retries: int = 100
):
    for _ in range(retries):
        try:
            with open(logspath) as logfile:
                if bool(re.search(log_to_search_for, logfile.read())):
                    return
        except FileNotFoundError:
            pass
        time.sleep(0.5)

    raise TimeoutError(
        f"Could not find the log '{log_to_search_for}' after '{retries}' tries.  Logs available at: '{logspath}'."
    )


class MonetDBSetupError(Exception):
    """Raised when the MonetDB container is unable to start."""


def _create_monetdb_container(cont_name, cont_port):
    print(f"\nCreating monetdb container '{cont_name}' at port {cont_port}...")
    client = docker.from_env()
    container_names = [container.name for container in client.containers.list(all=True)]
    if cont_name not in container_names:
        if PULL_DOCKER_IMAGES:
            print(f"\nPulling monetdb image '{TESTING_MONETDB_CONT_IMAGE}'.")
            client.images.pull(TESTING_MONETDB_CONT_IMAGE)
            print(f"\nPulled monetdb image '{TESTING_MONETDB_CONT_IMAGE}'.")
        # A volume is used to pass the udfio inside the monetdb container.
        # This is done so that we don't need to rebuild every time the udfio.py file is changed.
        udfio_full_path = path.abspath(udfio.__file__)
        container = client.containers.run(
            TESTING_MONETDB_CONT_IMAGE,
            detach=True,
            ports={"50000/tcp": cont_port},
            volumes=[
                f"{udfio_full_path}:/home/udflib/udfio.py",
                f"{TEST_DATA_FOLDER}:{TEST_DATA_FOLDER}",
            ],
            name=cont_name,
            publish_all_ports=True,
        )
    else:
        container = client.containers.get(cont_name)
        # After a machine restart the container exists, but it is stopped. (Used only in development)
        if container.status == "exited":
            container.start()

    # The time needed to start a monetdb container varies considerably. We need
    # to wait until a phrase appears in the logs to avoid starting the tests
    # too soon. The process is abandoned after 100 tries (50 sec).
    for _ in range(100):
        if b"new database mapi:monetdb" in container.logs():
            break
        time.sleep(0.5)
    else:
        raise MonetDBSetupError
    print(f"Monetdb container '{cont_name}' started.")


def restart_monetdb_container(cont_name):
    print(f"\nRestarting monetdb container '{cont_name}'.")
    client = docker.from_env()
    container = client.containers.get(cont_name)
    container.restart()
    # The time needed to restart a monetdb container varies considerably. We need
    # to wait until the phrase "attempting restart"
    # and then check for the "started 'db'" appears in the logs to avoid starting the tests
    # too soon. The process is abandoned after 100 tries (50 sec).
    for _ in range(100):
        container_logs = container.logs().decode("utf-8")
        if (
            "attempting restart" in container_logs
            and "started 'db'" in container_logs.split("attempting restart")[-1]
        ):
            break
        time.sleep(0.5)
    else:
        raise MonetDBSetupError

    print(f"Restarted monetdb container '{cont_name}'.")


def remove_monetdb_container(cont_name):
    print(f"\nRemoving monetdb container '{cont_name}'.")
    client = docker.from_env()
    container_names = [container.name for container in client.containers.list(all=True)]
    if cont_name in container_names:
        container = client.containers.get(cont_name)
        container.remove(v=True, force=True)
        print(f"Removed monetdb container '{cont_name}'.")
    else:
        print(f"Monetdb container '{cont_name}' is already removed.")


def get_worker_id(worker_config_file):
    worker_config_filepath = path.join(TEST_ENV_CONFIG_FOLDER, worker_config_file)
    with open(worker_config_filepath) as fp:
        tmp = toml.load(fp)
        return tmp["identifier"]


@pytest.fixture(scope="session")
def monetdb_globalworker():
    cont_name = MONETDB_GLOBALWORKER_NAME
    cont_port = MONETDB_GLOBALWORKER_PORT
    _create_monetdb_container(cont_name, cont_port)
    yield
    # TODO Very slow development if containers are always removed afterwards
    # _remove_monetdb_container(cont_name)


@pytest.fixture(scope="session")
def monetdb_localworker1():
    cont_name = MONETDB_LOCALWORKER1_NAME
    cont_port = MONETDB_LOCALWORKER1_PORT
    _create_monetdb_container(cont_name, cont_port)
    yield
    # TODO Very slow development if containers are always removed afterwards
    # _remove_monetdb_container(cont_name)


@pytest.fixture(scope="session")
def monetdb_localworker2():
    cont_name = MONETDB_LOCALWORKER2_NAME
    cont_port = MONETDB_LOCALWORKER2_PORT
    _create_monetdb_container(cont_name, cont_port)
    yield
    # TODO Very slow development if containers are always removed afterwards
    # _remove_monetdb_container(cont_name)


@pytest.fixture(scope="session")
def monetdb_smpc_globalworker():
    cont_name = MONETDB_SMPC_GLOBALWORKER_NAME
    cont_port = MONETDB_SMPC_GLOBALWORKER_PORT
    _create_monetdb_container(cont_name, cont_port)
    yield
    # TODO Very slow development if containers are always removed afterwards
    # _remove_monetdb_container(cont_name)


@pytest.fixture(scope="session")
def monetdb_smpc_localworker1():
    cont_name = MONETDB_SMPC_LOCALWORKER1_NAME
    cont_port = MONETDB_SMPC_LOCALWORKER1_PORT
    _create_monetdb_container(cont_name, cont_port)
    yield
    # TODO Very slow development if containers are always removed afterwards
    # _remove_monetdb_container(cont_name)


@pytest.fixture(scope="session")
def monetdb_smpc_localworker2():
    cont_name = MONETDB_SMPC_LOCALWORKER2_NAME
    cont_port = MONETDB_SMPC_LOCALWORKER2_PORT
    _create_monetdb_container(cont_name, cont_port)
    yield
    # TODO Very slow development if containers are always removed afterwards
    # _remove_monetdb_container(cont_name)


@pytest.fixture(scope="function")
def monetdb_localworkertmp():
    cont_name = MONETDB_LOCALWORKERTMP_NAME
    cont_port = MONETDB_LOCALWORKERTMP_PORT
    _create_monetdb_container(cont_name, cont_port)
    yield
    remove_monetdb_container(cont_name)


def _init_database_monetdb_container(db_port, worker_id):
    monetdb_configs = MonetDBConfigurations(db_port)
    print(f"\nInitializing database ({monetdb_configs.ip}:{monetdb_configs.port})")
    cmd = f"mipdb init --sqlite_db_path {TEST_DATA_FOLDER}/{worker_id}.db"
    subprocess.run(
        cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    print(f"\nDatabase ({monetdb_configs.ip}:{monetdb_configs.port}) initialized.")


def _load_test_data_monetdb_container(db_port, worker_id):
    monetdb_configs = MonetDBConfigurations(db_port)
    # Check if the database is already loaded
    cmd = f"mipdb list-datasets {monetdb_configs.convert_to_mipdb_format()} --sqlite_db_path {TEST_DATA_FOLDER}/{worker_id}.db"
    res = subprocess.run(
        cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if "There are no datasets" not in str(res.stdout):
        print(
            f"\nDatabase ({monetdb_configs.ip}:{monetdb_configs.port}) already loaded, continuing."
        )
        return

    datasets_per_data_model = {}
    # Load the test data folder into the dbs
    for dirpath, dirnames, filenames in os.walk(TEST_DATA_FOLDER):
        if "CDEsMetadata.json" not in filenames:
            continue
        cdes_file = os.path.join(dirpath, "CDEsMetadata.json")
        with open(cdes_file) as data_model_metadata_file:
            data_model_metadata = json.load(data_model_metadata_file)
            data_model_code = data_model_metadata["code"]
            data_model_version = data_model_metadata["version"]
            data_model = f"{data_model_code}:{data_model_version}"

        print(
            f"\nLoading data model '{data_model_code}:{data_model_version}' metadata to database ({monetdb_configs.ip}:{monetdb_configs.port})"
        )
        cmd = f"mipdb add-data-model {cdes_file} {monetdb_configs.convert_to_mipdb_format()} --sqlite_db_path {TEST_DATA_FOLDER}/{worker_id}.db"
        subprocess.run(
            cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        csvs = sorted(
            [f"{dirpath}/{file}" for file in filenames if file.endswith("test.csv")]
        )

        for csv in csvs:
            cmd = f"mipdb add-dataset {csv} -d {data_model_code} -v {data_model_version} {monetdb_configs.convert_to_mipdb_format()} --sqlite_db_path {TEST_DATA_FOLDER}/{worker_id}.db"
            subprocess.run(
                cmd,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            print(
                f"\nLoading dataset {pathlib.PurePath(csv).name} to database ({monetdb_configs.ip}:{monetdb_configs.port})"
            )
            datasets_per_data_model[data_model] = pathlib.PurePath(csv).name

    print(f"\nData loaded to database ({monetdb_configs.ip}:{monetdb_configs.port})")
    time.sleep(2)  # Needed to avoid db crash while loading
    return datasets_per_data_model


def _load_data_monetdb_container(db_port, dataset_suffixes, worker_id):
    monetdb_configs = MonetDBConfigurations(db_port)
    # Check if the database is already loaded
    cmd = f"mipdb list-datasets {monetdb_configs.convert_to_mipdb_format()} --sqlite_db_path {TEST_DATA_FOLDER}/{worker_id}.db"
    res = subprocess.run(
        cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if "There are no datasets" not in str(res.stdout):
        print(
            f"\nDatabase ({monetdb_configs.ip}:{monetdb_configs.port}) already loaded, continuing."
        )
        return

    datasets_per_data_model = {}
    # Load the test data folder into the dbs
    for dirpath, dirnames, filenames in os.walk(TEST_DATA_FOLDER):
        if "CDEsMetadata.json" not in filenames:
            continue
        cdes_file = os.path.join(dirpath, "CDEsMetadata.json")
        with open(cdes_file) as data_model_metadata_file:
            data_model_metadata = json.load(data_model_metadata_file)
            data_model_code = data_model_metadata["code"]
            data_model_version = data_model_metadata["version"]
            data_model = f"{data_model_code}:{data_model_version}"

        print(
            f"\nLoading data model '{data_model_code}:{data_model_version}' metadata to database ({monetdb_configs.ip}:{monetdb_configs.port})"
        )
        cmd = f"mipdb add-data-model {cdes_file} {monetdb_configs.convert_to_mipdb_format()} --sqlite_db_path {TEST_DATA_FOLDER}/{worker_id}.db"
        subprocess.run(
            cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        csvs = sorted(
            [
                f"{dirpath}/{file}"
                for file in filenames
                for suffix in dataset_suffixes
                if file.endswith(".csv") and str(suffix) in file
            ]
        )

        for csv in csvs:
            cmd = f"mipdb add-dataset {csv} -d {data_model_code} -v {data_model_version} {monetdb_configs.convert_to_mipdb_format()} --sqlite_db_path {TEST_DATA_FOLDER}/{worker_id}.db"
            subprocess.run(
                cmd,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            print(
                f"\nLoading dataset {pathlib.PurePath(csv).name} to database ({monetdb_configs.ip}:{monetdb_configs.port})"
            )
            datasets_per_data_model[data_model] = pathlib.PurePath(csv).name

    print(f"\nData loaded to database ({monetdb_configs.ip}:{monetdb_configs.port})")
    time.sleep(2)  # Needed to avoid db crash while loading
    return datasets_per_data_model


@pytest.fixture(scope="session")
def init_data_globalworker(monetdb_globalworker):
    worker_config_file = GLOBALWORKER_CONFIG_FILE
    worker_id = get_worker_id(worker_config_file)
    _init_database_monetdb_container(MONETDB_GLOBALWORKER_PORT, worker_id)
    yield


@pytest.fixture(scope="session")
def load_data_localworker1(monetdb_localworker1):
    worker_config_file = LOCALWORKER1_CONFIG_FILE
    worker_id = get_worker_id(worker_config_file)
    _init_database_monetdb_container(LOCALWORKER1_CONFIG_FILE, worker_id)
    loaded_datasets_per_data_model = _load_data_monetdb_container(
        MONETDB_LOCALWORKER1_PORT, DATASET_SUFFIXES_LOCALWORKER1, worker_id
    )
    yield loaded_datasets_per_data_model


@pytest.fixture(scope="session")
def load_data_localworker2(monetdb_localworker2):
    worker_config_file = LOCALWORKER2_CONFIG_FILE
    worker_id = get_worker_id(worker_config_file)
    _init_database_monetdb_container(MONETDB_LOCALWORKER2_PORT, worker_id)
    loaded_datasets_per_data_model = _load_data_monetdb_container(
        MONETDB_LOCALWORKER2_PORT, DATASET_SUFFIXES_LOCALWORKER2, worker_id
    )
    yield loaded_datasets_per_data_model


@pytest.fixture(scope="session")
def load_test_data_globalworker(monetdb_globalworker):
    worker_config_file = GLOBALWORKER_CONFIG_FILE
    worker_id = get_worker_id(worker_config_file)
    _init_database_monetdb_container(MONETDB_GLOBALWORKER_PORT, worker_id)
    loaded_datasets_per_data_model = _load_test_data_monetdb_container(
        MONETDB_GLOBALWORKER_PORT, worker_id
    )
    yield loaded_datasets_per_data_model


@pytest.fixture(scope="function")
def load_data_localworkertmp(monetdb_localworkertmp):
    worker_config_file = LOCALWORKERTMP_CONFIG_FILE
    worker_id = get_worker_id(worker_config_file)
    _init_database_monetdb_container(MONETDB_LOCALWORKERTMP_PORT, worker_id)
    loaded_datasets_per_data_model = _load_data_monetdb_container(
        MONETDB_LOCALWORKERTMP_PORT,
        DATASET_SUFFIXES_LOCALWORKERTMP,
        worker_id,
    )
    yield loaded_datasets_per_data_model


@pytest.fixture(scope="session")
def load_data_smpc_localworker1(monetdb_smpc_localworker1):
    worker_config_file = LOCALWORKER1_SMPC_CONFIG_FILE
    worker_id = get_worker_id(worker_config_file)
    _init_database_monetdb_container(MONETDB_SMPC_LOCALWORKER1_PORT, worker_id)
    loaded_datasets_per_data_model = _load_data_monetdb_container(
        MONETDB_SMPC_LOCALWORKER1_PORT,
        DATASET_SUFFIXES_SMPC_LOCALWORKER1,
        worker_id,
    )
    yield loaded_datasets_per_data_model


@pytest.fixture(scope="session")
def load_data_smpc_localworker2(monetdb_smpc_localworker2):
    worker_config_file = LOCALWORKER2_SMPC_CONFIG_FILE
    worker_id = get_worker_id(worker_config_file)
    _init_database_monetdb_container(MONETDB_SMPC_LOCALWORKER2_PORT, worker_id)
    loaded_datasets_per_data_model = _load_data_monetdb_container(
        MONETDB_SMPC_LOCALWORKER2_PORT,
        DATASET_SUFFIXES_SMPC_LOCALWORKER2,
        worker_id,
    )
    yield loaded_datasets_per_data_model


def _create_monetdb_cursor(db_port, db_username="executor", db_password="executor"):
    class MonetDBTesting:
        """MonetDB class used for testing."""

        def __init__(self) -> None:
            dbfarm = "db"
            url = (
                f"monetdb://{db_username}:{db_password}@{COMMON_IP}:{db_port}/{dbfarm}"
            )
            self._executor = sql.create_engine(url, echo=True)

        def execute(self, query, *args, **kwargs):
            return self._executor.execute(query, *args, **kwargs)

    return MonetDBTesting()


def _create_sqlite_cursor(worker_id):
    class SqliteDBTesting:
        """SqliteDBTesting class used for testing."""

        def __init__(self) -> None:
            self.url = f"{TEST_DATA_FOLDER}/{worker_id}.db"

        def execute(self, query):
            conn = sqlite3.connect(f"{TEST_DATA_FOLDER}/{worker_id}.db")
            cur = conn.cursor()
            cur.execute(query)
            cur.fetchall()
            cur.close()
            conn.commit()
            conn.close()

    return SqliteDBTesting()


@pytest.fixture(scope="session")
def globalworker_sqlite_db_cursor():
    return _create_sqlite_cursor("testglobalworker")


@pytest.fixture(scope="session")
def globalworker_db_cursor():
    return _create_monetdb_cursor(MONETDB_GLOBALWORKER_PORT)


@pytest.fixture(scope="session")
def localworker1_db_cursor():
    return _create_monetdb_cursor(MONETDB_LOCALWORKER1_PORT)


@pytest.fixture(scope="session")
def localworker2_db_cursor():
    return _create_monetdb_cursor(MONETDB_LOCALWORKER2_PORT)


@pytest.fixture(scope="session")
def globalworker_smpc_db_cursor():
    return _create_monetdb_cursor(MONETDB_SMPC_GLOBALWORKER_PORT)


@pytest.fixture(scope="session")
def localworker1_smpc_db_cursor():
    return _create_monetdb_cursor(MONETDB_SMPC_LOCALWORKER1_PORT)


@pytest.fixture(scope="session")
def localworker2_smpc_db_cursor():
    return _create_monetdb_cursor(MONETDB_SMPC_LOCALWORKER2_PORT)


@pytest.fixture(scope="function")
def localworkertmp_db_cursor():
    return _create_monetdb_cursor(MONETDB_LOCALWORKERTMP_PORT)


def create_table_in_db(
    db_cursor,
    table_name: str,
    table_schema: TableSchema,
    publish_table: bool = False,
):
    query_schema = ",".join(
        [f"{column.name} {column.dtype.to_sql()}" for column in table_schema.columns]
    )
    create_table_query = f"CREATE TABLE {table_name} ({query_schema});"
    publish_table_query = (
        f"GRANT SELECT ON TABLE {table_name} TO guest;" if publish_table else ""
    )
    db_cursor.run(create_table_query + publish_table_query)


def insert_data_to_db(
    table_name: str, table_values: List[List[Union[str, int, float]]], db_cursor
):
    row_length = len(table_values[0])
    if all(len(row) != row_length for row in table_values):
        raise Exception("Not all rows have the same number of values")

    values = ", ".join(
        "(" + ", ".join("%s" for _ in range(row_length)) + ")" for _ in table_values
    )
    sql_clause = f"INSERT INTO {table_name} VALUES {values}"

    db_cursor.run(sql_clause, list(chain(*table_values)))


def get_table_data_from_db(
    db_cursor,
    table_name: str,
):
    return db_cursor.run(f"SELECT * FROM {table_name};").fetchall()


def _clean_db(cursor):
    class TableType(enum.Enum):
        NORMAL = 0
        VIEW = 1
        MERGE = 3
        REMOTE = 5

    # Order of the table types matter not to have dependencies when dropping the tables
    table_type_drop_order = (
        TableType.MERGE,
        TableType.REMOTE,
        TableType.VIEW,
        TableType.NORMAL,
    )
    for table_type in table_type_drop_order:
        select_user_tables = f"SELECT name FROM sys.tables WHERE system=FALSE AND schema_id  in (SELECT id from schemas where system=false and name='executor') AND type={table_type.value}"
        user_tables = cursor.run(select_user_tables).fetchall()
        for table_name, *_ in user_tables:
            if table_type == TableType.VIEW:
                cursor.run(f"DROP VIEW {table_name}")
            else:
                cursor.run(f"DROP TABLE {table_name}")


@pytest.fixture(scope="function")
def schedule_clean_globalworker_db(globalworker_db_cursor):
    yield
    _clean_db(globalworker_db_cursor)


@pytest.fixture(scope="function")
def schedule_clean_localworker1_db(localworker1_db_cursor):
    yield
    _clean_db(localworker1_db_cursor)


@pytest.fixture(scope="function")
def schedule_clean_smpc_globalworker_db(globalworker_smpc_db_cursor):
    yield
    _clean_db(globalworker_smpc_db_cursor)


@pytest.fixture(scope="function")
def schedule_clean_smpc_localworker1_db(localworker1_smpc_db_cursor):
    yield
    _clean_db(localworker1_smpc_db_cursor)


@pytest.fixture(scope="function")
def schedule_clean_smpc_localworker2_db(localworker2_smpc_db_cursor):
    yield
    _clean_db(localworker2_smpc_db_cursor)


@pytest.fixture(scope="function")
def schedule_clean_localworker2_db(localworker2_db_cursor):
    yield
    _clean_db(localworker2_db_cursor)


@pytest.fixture(scope="function")
def use_globalworker_database(monetdb_globalworker, schedule_clean_globalworker_db):
    pass


@pytest.fixture(scope="function")
def use_localworker1_database(monetdb_localworker1, schedule_clean_localworker1_db):
    pass


@pytest.fixture(scope="function")
def use_localworker2_database(monetdb_localworker2, schedule_clean_localworker2_db):
    pass


@pytest.fixture(scope="function")
def use_smpc_globalworker_database(
    monetdb_smpc_globalworker, schedule_clean_smpc_globalworker_db
):
    pass


@pytest.fixture(scope="function")
def use_smpc_localworker1_database(
    monetdb_smpc_localworker1, schedule_clean_smpc_localworker1_db
):
    pass


@pytest.fixture(scope="function")
def use_smpc_localworker2_database(
    monetdb_smpc_localworker2, schedule_clean_smpc_localworker2_db
):
    pass


def _create_rabbitmq_container(cont_name, cont_port):
    print(f"\nCreating rabbitmq container '{cont_name}' at port {cont_port}...")
    client = docker.from_env()
    container_names = [container.name for container in client.containers.list(all=True)]
    if cont_name not in container_names:
        if PULL_DOCKER_IMAGES:
            print(f"\nPulling rabbitmq image '{TESTING_RABBITMQ_CONT_IMAGE}'.")
            client.images.pull(TESTING_RABBITMQ_CONT_IMAGE)
            print(f"\nPulled rabbitmq image '{TESTING_RABBITMQ_CONT_IMAGE}'.")

        container = client.containers.run(
            TESTING_RABBITMQ_CONT_IMAGE,
            detach=True,
            ports={"5672/tcp": cont_port, "15672/tcp": cont_port + 100},
            name=cont_name,
        )
    else:
        container = client.containers.get(cont_name)
        # After a machine restart the container exists, but it is stopped. (Used only in development)
        if container.status == "exited":
            container.start()

    while (
        "Health" not in container.attrs["State"]
        or container.attrs["State"]["Health"]["Status"] != "healthy"
    ):
        container.reload()  # attributes are cached, this refreshes them..
        time.sleep(1)

    print(f"Rabbitmq container '{cont_name}' started.")


def _remove_rabbitmq_container(cont_name):
    print(f"\nRemoving rabbitmq container '{cont_name}'.")
    try:
        client = docker.from_env()
        container = client.containers.get(cont_name)
        container.remove(v=True, force=True)
    except docker.errors.NotFound:
        print(
            f"(conftest.py::_remove_rabbitmq_container) container {cont_name=} was not "
            f"found, probably already removed"
        )
        pass  # container was removed by other means...
    print(f"Removed rabbitmq container '{cont_name}'.")


@pytest.fixture(scope="session")
def rabbitmq_globalworker():
    cont_name = RABBITMQ_GLOBALWORKER_NAME
    cont_port = RABBITMQ_GLOBALWORKER_PORT
    _create_rabbitmq_container(cont_name, cont_port)
    yield
    # TODO Very slow development if containers are always removed afterwards
    # _remove_rabbitmq_container(cont_name)


@pytest.fixture(scope="session")
def rabbitmq_localworker1():
    cont_name = RABBITMQ_LOCALWORKER1_NAME
    cont_port = RABBITMQ_LOCALWORKER1_PORT
    _create_rabbitmq_container(cont_name, cont_port)
    yield
    # TODO Very slow development if containers are always removed afterwards
    # _remove_rabbitmq_container(cont_name)


@pytest.fixture(scope="session")
def rabbitmq_localworker2():
    cont_name = RABBITMQ_LOCALWORKER2_NAME
    cont_port = RABBITMQ_LOCALWORKER2_PORT
    _create_rabbitmq_container(cont_name, cont_port)
    yield
    # TODO Very slow development if containers are always removed afterwards
    # _remove_rabbitmq_container(cont_name)


@pytest.fixture(scope="session")
def rabbitmq_smpc_globalworker():
    cont_name = RABBITMQ_SMPC_GLOBALWORKER_NAME
    cont_port = RABBITMQ_SMPC_GLOBALWORKER_PORT
    _create_rabbitmq_container(cont_name, cont_port)
    yield
    # TODO Very slow development if containers are always removed afterwards
    # _remove_rabbitmq_container(cont_name)


@pytest.fixture(scope="session")
def rabbitmq_smpc_localworker1():
    cont_name = RABBITMQ_SMPC_LOCALWORKER1_NAME
    cont_port = RABBITMQ_SMPC_LOCALWORKER1_PORT
    _create_rabbitmq_container(cont_name, cont_port)
    yield
    # TODO Very slow development if containers are always removed afterwards
    # _remove_rabbitmq_container(cont_name)


@pytest.fixture(scope="session")
def rabbitmq_smpc_localworker2():
    cont_name = RABBITMQ_SMPC_LOCALWORKER2_NAME
    cont_port = RABBITMQ_SMPC_LOCALWORKER2_PORT
    _create_rabbitmq_container(cont_name, cont_port)
    yield
    # TODO Very slow development if containers are always removed afterwards
    # _remove_rabbitmq_container(cont_name)


@pytest.fixture(scope="function")
def rabbitmq_localworkertmp():
    cont_name = RABBITMQ_LOCALWORKERTMP_NAME
    cont_port = RABBITMQ_LOCALWORKERTMP_PORT
    _create_rabbitmq_container(cont_name, cont_port)
    yield
    _remove_rabbitmq_container(cont_name)


def remove_localworkertmp_rabbitmq():
    cont_name = RABBITMQ_LOCALWORKERTMP_NAME
    _remove_rabbitmq_container(cont_name)


def _create_aggregation_server_service(
    aggregation_server_config_filepath: str,
    logs_filename: str = "test_aggregation_server.out",
):
    """
    Launches aggregation_server.server as a subprocess, capturing logs.
    """
    logpath = OUTDIR / logs_filename
    if logpath.exists():
        logpath.unlink()

    env = os.environ.copy()
    env["AGG_SERVER_CONFIG_FILE"] = str(aggregation_server_config_filepath)

    # Directly run the module—no Poetry wrapper
    cmd = f"exec python -m aggregation_server.server >> {logpath} 2>&1"

    print(f"\nStarting aggregation_server (logs → {logpath})…")
    proc = subprocess.Popen(cmd, shell=True, env=env)

    # Wait for the startup message in the log (timeout in seconds)
    _search_for_string_in_logfile("Aggregation server running", logpath)

    print(f"[TEST] aggregation_server is up (PID {proc.pid}).")
    return proc


@pytest.fixture(scope="module")
def aggregation_server_service():
    """
    Pytest fixture: starts the aggregation_server for all tests in the module,
    then ensures it’s cleaned up at teardown.
    """
    aggregation_server_config_filepath = path.join(
        TEST_ENV_CONFIG_FOLDER, AGG_SERVER_CONFIG_FILE
    )
    proc = _create_aggregation_server_service(aggregation_server_config_filepath)
    # allow a brief warm-up
    time.sleep(0.5)
    yield proc
    kill_service(proc)


def _create_worker_service(worker_config_filepath):
    with open(worker_config_filepath) as fp:
        tmp = toml.load(fp)
        worker_id = tmp["identifier"]

    print(f"\nCreating worker service with id '{worker_id}'...")

    logpath = OUTDIR / (worker_id + ".out")
    if os.path.isfile(logpath):
        os.remove(logpath)

    env = os.environ.copy()
    env["EXAREME2_ALGORITHM_FOLDERS"] = EXAREME2_ALGORITHM_FOLDERS_ENV_VARIABLE_VALUE
    env["FLOWER_ALGORITHM_FOLDERS"] = FLOWER_ALGORITHM_FOLDERS_ENV_VARIABLE_VALUE
    env["EXAFLOW_ALGORITHM_FOLDERS"] = EXAFLOW_ALGORITHM_FOLDERS_ENV_VARIABLE_VALUE
    env["DATA_PATH"] = str(TEST_DATA_FOLDER)
    env["EXAREME2_WORKER_CONFIG_FILE"] = worker_config_filepath

    cmd = f"poetry run celery -A exareme2.worker.utils.celery_app worker -l  DEBUG >> {logpath}  --pool=eventlet --purge 2>&1 "

    # if executed without "exec" it is spawned as a child process of the shell, so it is difficult to kill it
    # https://stackoverflow.com/questions/4789837/how-to-terminate-a-python-subprocess-launched-with-shell-true
    proc = subprocess.Popen(
        "exec " + cmd,
        shell=True,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        env=env,
    )

    # Check that celery started
    _search_for_string_in_logfile("celery@.* ready.", logpath)

    print(f"Created worker service with id '{worker_id}' and process id '{proc.pid}'.")
    return proc


def kill_service(proc):
    print(f"\nKilling service with process id '{proc.pid}'...")
    psutil_proc = psutil.Process(proc.pid)

    # First killing all the subprocesses, if they exist
    for child in psutil.Process(proc.pid).children(recursive=True):
        child.kill()
    proc.kill()

    for _ in range(100):
        if psutil_proc.status() == "zombie" or psutil_proc.status() == "sleeping":
            break
        time.sleep(0.1)
    else:
        raise TimeoutError(
            f"Service is still running, status: '{psutil_proc.status()}'."
        )
    print(f"Killed service with process id '{proc.pid}'.")


@pytest.fixture(scope="session")
def globalworker_worker_service(rabbitmq_globalworker, monetdb_globalworker):
    worker_config_file = GLOBALWORKER_CONFIG_FILE
    worker_config_filepath = path.join(TEST_ENV_CONFIG_FOLDER, worker_config_file)
    proc = _create_worker_service(worker_config_filepath)
    yield
    kill_service(proc)


@pytest.fixture(scope="session")
def localworker1_worker_service(rabbitmq_localworker1, monetdb_localworker1):
    worker_config_file = LOCALWORKER1_CONFIG_FILE
    worker_config_filepath = path.join(TEST_ENV_CONFIG_FOLDER, worker_config_file)
    proc = _create_worker_service(worker_config_filepath)
    yield
    kill_service(proc)


@pytest.fixture(scope="session")
def localworker2_worker_service(rabbitmq_localworker2, monetdb_localworker2):
    worker_config_file = LOCALWORKER2_CONFIG_FILE
    worker_config_filepath = path.join(TEST_ENV_CONFIG_FOLDER, worker_config_file)
    proc = _create_worker_service(worker_config_filepath)
    yield
    kill_service(proc)


@pytest.fixture(scope="session")
def smpc_globalworker_worker_service(
    rabbitmq_smpc_globalworker, monetdb_smpc_globalworker
):
    worker_config_file = GLOBALWORKER_SMPC_CONFIG_FILE
    worker_config_filepath = path.join(TEST_ENV_CONFIG_FOLDER, worker_config_file)
    proc = _create_worker_service(worker_config_filepath)
    yield
    kill_service(proc)


@pytest.fixture(scope="session")
def smpc_localworker1_worker_service(
    rabbitmq_smpc_localworker1, monetdb_smpc_localworker1
):
    worker_config_file = LOCALWORKER1_SMPC_CONFIG_FILE
    worker_config_filepath = path.join(TEST_ENV_CONFIG_FOLDER, worker_config_file)
    proc = _create_worker_service(worker_config_filepath)
    yield
    kill_service(proc)


@pytest.fixture(scope="session")
def smpc_localworker2_worker_service(
    rabbitmq_smpc_localworker2, monetdb_smpc_localworker2
):
    worker_config_file = LOCALWORKER2_SMPC_CONFIG_FILE
    worker_config_filepath = path.join(TEST_ENV_CONFIG_FOLDER, worker_config_file)
    proc = _create_worker_service(worker_config_filepath)
    yield
    kill_service(proc)


def create_localworkertmp_worker_service():
    worker_config_file = LOCALWORKERTMP_CONFIG_FILE
    worker_config_filepath = path.join(TEST_ENV_CONFIG_FOLDER, worker_config_file)
    return _create_worker_service(worker_config_filepath)


@pytest.fixture(scope="function")
def localworkertmp_worker_service(rabbitmq_localworkertmp, monetdb_localworkertmp):
    """
    ATTENTION!
    This worker service fixture is the only one returning the process, so it can be killed.
    The scope of the fixture is function, so it won't break tests if the worker service is killed.
    The rabbitmq and monetdb containers have also 'function' scope so this is VERY slow.
    This should be used only when the service should be killed e.g. for testing.
    """
    proc = create_localworkertmp_worker_service()
    yield proc
    kill_service(proc)


def is_localworkertmp_worker_service_ok(worker_process):
    psutil_proc = psutil.Process(worker_process.pid)
    return psutil_proc.status() != "zombie" and psutil_proc.status() != "sleeping"


def create_exareme2_tasks_handler_celery(worker_config_filepath):
    with open(worker_config_filepath) as fp:
        tmp = toml.load(fp)
        worker_id = tmp["identifier"]
        queue_domain = tmp["rabbitmq"]["ip"]
        queue_port = tmp["rabbitmq"]["port"]
        db_domain = tmp["monetdb"]["ip"]
        db_port = tmp["monetdb"]["port"]
    queue_address = ":".join([str(queue_domain), str(queue_port)])
    db_address = ":".join([str(db_domain), str(db_port)])

    return Exareme2TasksHandler(
        request_id=REQUEST_ID,
        worker_id=worker_id,
        worker_queue_addr=queue_address,
        worker_db_addr=db_address,
        tasks_timeout=TASKS_TIMEOUT,
        run_udf_task_timeout=RUN_UDF_TASK_TIMEOUT,
    )


@pytest.fixture(scope="session")
def globalworker_tasks_handler(globalworker_worker_service):
    worker_config_filepath = path.join(TEST_ENV_CONFIG_FOLDER, GLOBALWORKER_CONFIG_FILE)
    tasks_handler = create_exareme2_tasks_handler_celery(worker_config_filepath)
    return tasks_handler


@pytest.fixture(scope="session")
def localworker1_tasks_handler(localworker1_worker_service):
    worker_config_filepath = path.join(TEST_ENV_CONFIG_FOLDER, LOCALWORKER1_CONFIG_FILE)
    tasks_handler = create_exareme2_tasks_handler_celery(worker_config_filepath)
    return tasks_handler


@pytest.fixture(scope="session")
def localworker2_tasks_handler(localworker2_worker_service):
    worker_config_filepath = path.join(TEST_ENV_CONFIG_FOLDER, LOCALWORKER2_CONFIG_FILE)
    tasks_handler = create_exareme2_tasks_handler_celery(worker_config_filepath)
    return tasks_handler


@pytest.fixture(scope="function")
def localworkertmp_tasks_handler(localworkertmp_worker_service):
    worker_config_filepath = path.join(
        TEST_ENV_CONFIG_FOLDER, LOCALWORKERTMP_CONFIG_FILE
    )
    tasks_handler = create_exareme2_tasks_handler_celery(worker_config_filepath)
    return tasks_handler


def get_worker_config_by_id(worker_config_file: str):
    with open(path.join(TEST_ENV_CONFIG_FOLDER, worker_config_file)) as fp:
        worker_config = AttrDict(toml.load(fp))
    return worker_config


@pytest.fixture(scope="function")
def globalworker_celery_app(globalworker_worker_service):
    return CeleryAppFactory().get_celery_app(socket_addr=RABBITMQ_GLOBALWORKER_ADDR)


@pytest.fixture(scope="function")
def localworker1_celery_app(localworker1_worker_service):
    return CeleryAppFactory().get_celery_app(socket_addr=RABBITMQ_LOCALWORKER1_ADDR)


@pytest.fixture(scope="function")
def localworker2_celery_app(localworker2_worker_service):
    return CeleryAppFactory().get_celery_app(socket_addr=RABBITMQ_LOCALWORKER2_ADDR)


@pytest.fixture(scope="function")
def localworkertmp_celery_app(localworkertmp_worker_service):
    return CeleryAppFactory().get_celery_app(socket_addr=RABBITMQ_LOCALWORKERTMP_ADDR)


@pytest.fixture(scope="function")
def smpc_globalworker_celery_app(smpc_globalworker_worker_service):
    return CeleryAppFactory().get_celery_app(
        socket_addr=RABBITMQ_SMPC_GLOBALWORKER_ADDR
    )


@pytest.fixture(scope="function")
def smpc_localworker1_celery_app(smpc_localworker1_worker_service):
    return CeleryAppFactory().get_celery_app(
        socket_addr=RABBITMQ_SMPC_LOCALWORKER1_ADDR
    )


@pytest.fixture(scope="session")
def smpc_localworker2_celery_app(smpc_localworker2_worker_service):
    return CeleryAppFactory().get_celery_app(
        socket_addr=RABBITMQ_SMPC_LOCALWORKER2_ADDR
    )


@pytest.fixture(scope="function")
def reset_celery_app_factory():
    CeleryAppFactory()._celery_apps = {}


@pytest.fixture(scope="function")
def controller_service_with_localworkertmp():
    service_port = CONTROLLER_PORT
    controller_config_filepath = path.join(
        TEST_ENV_CONFIG_FOLDER, CONTROLLER_CONFIG_FILE
    )
    localworkers_config_filepath = path.join(
        TEST_ENV_CONFIG_FOLDER, CONTROLLER_LOCALWORKERTMP_ADDRESSES_FILE
    )

    proc = _create_controller_service(
        service_port,
        controller_config_filepath,
        localworkers_config_filepath,
        CONTROLLER_OUTPUT_FILE,
    )
    yield
    kill_service(proc)


@pytest.fixture(scope="function")
def controller_service_with_localworker1():
    service_port = CONTROLLER_PORT
    controller_config_filepath = path.join(
        TEST_ENV_CONFIG_FOLDER, CONTROLLER_CONFIG_FILE
    )
    localworkers_config_filepath = path.join(
        TEST_ENV_CONFIG_FOLDER, CONTROLLER_GLOBALWORKER_LOCALWORKER1_ADDRESSES_FILE
    )

    proc = _create_controller_service(
        service_port,
        controller_config_filepath,
        localworkers_config_filepath,
        CONTROLLER_OUTPUT_FILE,
    )
    yield
    kill_service(proc)


@pytest.fixture(scope="function")
def controller_service_with_localworker_1_2():
    service_port = CONTROLLER_PORT
    controller_config_filepath = path.join(
        TEST_ENV_CONFIG_FOLDER, CONTROLLER_CONFIG_FILE
    )
    localworkers_config_filepath = path.join(
        TEST_ENV_CONFIG_FOLDER,
        CONTROLLER_GLOBALWORKER_LOCALWORKER1_LOCALWORKER2_ADDRESSES_FILE,
    )

    proc = _create_controller_service(
        service_port,
        controller_config_filepath,
        localworkers_config_filepath,
        CONTROLLER_OUTPUT_FILE,
    )
    yield
    kill_service(proc)


@pytest.fixture(scope="function")
def smpc_controller_service():
    service_port = CONTROLLER_SMPC_PORT
    controller_config_filepath = path.join(
        TEST_ENV_CONFIG_FOLDER, CONTROLLER_SMPC_CONFIG_FILE
    )
    localworkers_config_filepath = path.join(
        TEST_ENV_CONFIG_FOLDER, CONTROLLER_SMPC_LOCALWORKERS_CONFIG_FILE
    )

    proc = _create_controller_service(
        service_port,
        controller_config_filepath,
        localworkers_config_filepath,
        SMPC_CONTROLLER_OUTPUT_FILE,
    )
    yield proc
    kill_service(proc)


def smpc_controller_service_with_dp():
    service_port = CONTROLLER_SMPC_PORT
    controller_config_filepath = path.join(
        TEST_ENV_CONFIG_FOLDER, CONTROLLER_SMPC_DP_CONFIG_FILE
    )
    localworkers_config_filepath = path.join(
        TEST_ENV_CONFIG_FOLDER, CONTROLLER_SMPC_LOCALWORKERS_CONFIG_FILE
    )

    proc = _create_controller_service(
        service_port,
        controller_config_filepath,
        localworkers_config_filepath,
        SMPC_CONTROLLER_OUTPUT_FILE,
    )
    return proc


def _create_controller_service(
    service_port: int,
    controller_config_filepath: str,
    localworkers_config_filepath: str,
    logs_filename: str,
):
    print(f"\nCreating controller service on port '{service_port}'...")

    logpath = OUTDIR / logs_filename
    if os.path.isfile(logpath):
        os.remove(logpath)

    env = os.environ.copy()
    env["EXAREME2_ALGORITHM_FOLDERS"] = EXAREME2_ALGORITHM_FOLDERS_ENV_VARIABLE_VALUE
    env["FLOWER_ALGORITHM_FOLDERS"] = FLOWER_ALGORITHM_FOLDERS_ENV_VARIABLE_VALUE
    env["EXAFLOW_ALGORITHM_FOLDERS"] = EXAFLOW_ALGORITHM_FOLDERS_ENV_VARIABLE_VALUE
    env["LOCALWORKERS_CONFIG_FILE"] = localworkers_config_filepath
    env["EXAREME2_CONTROLLER_CONFIG_FILE"] = controller_config_filepath
    env["PYTHONPATH"] = str(Path(__file__).parent.parent.parent)

    cmd = f"poetry run hypercorn --config python:exareme2.controller.quart.hypercorn_config -b 0.0.0.0:{service_port} exareme2/controller/quart/app:app >> {logpath} 2>&1 "

    # if executed without "exec" it is spawned as a child process of the shell, so it is difficult to kill it
    # https://stackoverflow.com/questions/4789837/how-to-terminate-a-python-subprocess-launched-with-shell-true
    proc = subprocess.Popen(
        "exec " + cmd,
        shell=True,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        env=env,
    )

    # Check that hypercorn started
    _search_for_string_in_logfile("Running on", logpath)

    # Check that workers were loaded
    _search_for_string_in_logfile("Workers:", logpath)
    print(f"\nCreated controller service on port '{service_port}'.")

    return proc


@pytest.fixture(scope="session")
def smpc_coordinator():
    if USE_EXTERNAL_SMPC_CLUSTER:
        print(f"\nUsing external smpc cluster. smpc coordinator won't be started.")
        yield
        return

    docker_cli = docker.from_env()

    print(f"\nWaiting for smpc coordinator db to be ready...")
    # Start coordinator db
    try:
        docker_cli.containers.get(SMPC_COORD_DB_CONT_NAME)
    except docker.errors.NotFound:
        docker_cli.containers.run(
            image=SMPC_COORD_DB_IMAGE,
            name=SMPC_COORD_DB_CONT_NAME,
            detach=True,
            ports={27017: SMPC_COORD_DB_PORT},
            environment={
                "MONGO_INITDB_ROOT_USERNAME": "sysadmin",
                "MONGO_INITDB_ROOT_PASSWORD": "123qwe",
            },
        )
    print("Created coordinator db service.")

    # Start coordinator queue
    print(f"\nWaiting for smpc coordinator queue to be ready...")
    try:
        docker_cli.containers.get(SMPC_COORD_QUEUE_CONT_NAME)
    except docker.errors.NotFound:
        docker_cli.containers.run(
            image=SMPC_COORD_QUEUE_IMAGE,
            name=SMPC_COORD_QUEUE_CONT_NAME,
            detach=True,
            ports={6379: SMPC_COORD_QUEUE_PORT},
            environment={
                "REDIS_REPLICATION_MODE": "master",
            },
            command="redis-server --requirepass agora",
        )
    print("Created coordinator queue service.")

    # Start coordinator
    print(f"\nWaiting for smpc coordinator to be ready...")
    try:
        docker_cli.containers.get(SMPC_COORD_CONT_NAME)
    except docker.errors.NotFound:
        docker_cli.containers.run(
            image=SMPC_CLUSTER_IMAGE,
            name=SMPC_COORD_CONT_NAME,
            detach=True,
            ports={12314: SMPC_COORD_PORT},
            environment={
                "PLAYER_REPO_0": f"http://{COMMON_IP}:{SMPC_PLAYER1_PORT2}",
                "PLAYER_REPO_1": f"http://{COMMON_IP}:{SMPC_PLAYER2_PORT2}",
                "PLAYER_REPO_2": f"http://{COMMON_IP}:{SMPC_PLAYER3_PORT2}",
                "REDIS_HOST": f"{COMMON_IP}",
                "REDIS_PORT": f"{SMPC_COORD_QUEUE_PORT}",
                "REDIS_PSWD": "agora",
                "DB_URL": f"{COMMON_IP}:{SMPC_COORD_DB_PORT}",
                "DB_UNAME": "sysadmin",
                "DB_PSWD": "123qwe",
            },
            command="python coordinator.py",
        )
    print("Created coordinator service.")

    yield

    # TODO Very slow development if containers are always removed afterwards
    # db_cont = docker_cli.containers.get(SMPC_COORD_DB_CONT_NAME)
    # db_cont.remove(v=True, force=True)
    # queue_cont = docker_cli.containers.get(SMPC_COORD_QUEUE_CONT_NAME)
    # queue_cont.remove(v=True, force=True)
    # coord_cont = docker_cli.containers.get(SMPC_COORD_CONT_NAME)
    # coord_cont.remove(v=True, force=True)


@pytest.fixture(scope="session")
def smpc_players():
    if USE_EXTERNAL_SMPC_CLUSTER:
        print(f"\nUsing external smpc cluster. smpc players won't be started.")
        yield
        return

    docker_cli = docker.from_env()

    # Start player 1
    print(f"\nWaiting for smpc player 1 to be ready...")
    try:
        docker_cli.containers.get(SMPC_PLAYER1_CONT_NAME)
    except docker.errors.NotFound:
        docker_cli.containers.run(
            image=SMPC_CLUSTER_IMAGE,
            name=SMPC_PLAYER1_CONT_NAME,
            detach=True,
            ports={
                6000: SMPC_PLAYER1_PORT1,
                7000: SMPC_PLAYER1_PORT2,
                14000: SMPC_PLAYER1_PORT3,
            },
            environment={
                "PLAYER_REPO_0": f"http://{COMMON_IP}:{SMPC_PLAYER1_PORT2}",
                "PLAYER_REPO_1": f"http://{COMMON_IP}:{SMPC_PLAYER2_PORT2}",
                "PLAYER_REPO_2": f"http://{COMMON_IP}:{SMPC_PLAYER3_PORT2}",
                "COORDINATOR_URL": f"http://{COMMON_IP}:{SMPC_COORD_PORT}",
                "DB_URL": f"{COMMON_IP}:{SMPC_COORD_DB_PORT}",
                "DB_UNAME": "sysadmin",
                "DB_PSWD": "123qwe",
                "PORT": f"{SMPC_PLAYER1_PORT2}",
            },
            command="python player.py 0",
        )
    print("Created smpc player 1 service.")

    # Start player 2
    print(f"\nWaiting for smpc player 2 to be ready...")
    try:
        docker_cli.containers.get(SMPC_PLAYER2_CONT_NAME)
    except docker.errors.NotFound:
        docker_cli.containers.run(
            image=SMPC_CLUSTER_IMAGE,
            name=SMPC_PLAYER2_CONT_NAME,
            detach=True,
            ports={
                6001: SMPC_PLAYER2_PORT1,
                7001: SMPC_PLAYER2_PORT2,
                14001: SMPC_PLAYER2_PORT3,
            },
            environment={
                "PLAYER_REPO_0": f"http://{COMMON_IP}:{SMPC_PLAYER1_PORT2}",
                "PLAYER_REPO_1": f"http://{COMMON_IP}:{SMPC_PLAYER2_PORT2}",
                "PLAYER_REPO_2": f"http://{COMMON_IP}:{SMPC_PLAYER3_PORT2}",
                "COORDINATOR_URL": f"http://{COMMON_IP}:{SMPC_COORD_PORT}",
                "DB_URL": f"{COMMON_IP}:{SMPC_COORD_DB_PORT}",
                "DB_UNAME": "sysadmin",
                "DB_PSWD": "123qwe",
                "PORT": f"{SMPC_PLAYER2_PORT2}",
            },
            command="python player.py 1",
        )
    print("Created smpc player 2 service.")

    # Start player 3
    print(f"\nWaiting for smpc player 3 to be ready...")
    try:
        docker_cli.containers.get(SMPC_PLAYER3_CONT_NAME)
    except docker.errors.NotFound:
        docker_cli.containers.run(
            image=SMPC_CLUSTER_IMAGE,
            name=SMPC_PLAYER3_CONT_NAME,
            detach=True,
            ports={
                6002: SMPC_PLAYER3_PORT1,
                7002: SMPC_PLAYER3_PORT2,
                14002: SMPC_PLAYER3_PORT3,
            },
            environment={
                "PLAYER_REPO_0": f"http://{COMMON_IP}:{SMPC_PLAYER1_PORT2}",
                "PLAYER_REPO_1": f"http://{COMMON_IP}:{SMPC_PLAYER2_PORT2}",
                "PLAYER_REPO_2": f"http://{COMMON_IP}:{SMPC_PLAYER3_PORT2}",
                "COORDINATOR_URL": f"http://{COMMON_IP}:{SMPC_COORD_PORT}",
                "DB_URL": f"{COMMON_IP}:{SMPC_COORD_DB_PORT}",
                "DB_UNAME": "sysadmin",
                "DB_PSWD": "123qwe",
                "PORT": f"{SMPC_PLAYER3_PORT2}",
            },
            command="python player.py 2",
        )
    print("Created smpc player 3 service.")

    yield

    # TODO Very slow development if containers are always removed afterwards
    # player1_cont = docker_cli.containers.get(SMPC_PLAYER1_CONT_NAME)
    # player1_cont.remove(v=True, force=True)
    # player2_cont = docker_cli.containers.get(SMPC_PLAYER2_CONT_NAME)
    # player2_cont.remove(v=True, force=True)
    # player3_cont = docker_cli.containers.get(SMPC_PLAYER3_CONT_NAME)
    # player3_cont.remove(v=True, force=True)


@pytest.fixture(scope="session")
def smpc_clients():
    if USE_EXTERNAL_SMPC_CLUSTER:
        print(f"\nUsing external smpc cluster. smpc clients won't be started.")
        yield
        return

    docker_cli = docker.from_env()

    # Start client 1
    print(f"\nWaiting for smpc client 1 to be ready...")
    try:
        docker_cli.containers.get(SMPC_CLIENT1_CONT_NAME)
    except docker.errors.NotFound:
        with open(
            path.join(TEST_ENV_CONFIG_FOLDER, LOCALWORKER1_SMPC_CONFIG_FILE)
        ) as fp:
            tmp = toml.load(fp)
            client_id = tmp["smpc"]["client_id"]
        docker_cli.containers.run(
            image=SMPC_CLUSTER_IMAGE,
            name=SMPC_CLIENT1_CONT_NAME,
            detach=True,
            ports={
                SMPC_CLIENT1_PORT: SMPC_CLIENT1_PORT,
            },
            environment={
                "PLAYER_REPO_0": f"http://{COMMON_IP}:{SMPC_PLAYER1_PORT2}",
                "PLAYER_REPO_1": f"http://{COMMON_IP}:{SMPC_PLAYER2_PORT2}",
                "PLAYER_REPO_2": f"http://{COMMON_IP}:{SMPC_PLAYER3_PORT2}",
                "COORDINATOR_URL": f"http://{COMMON_IP}:{SMPC_COORD_PORT}",
                "ID": client_id,
                "PORT": f"{SMPC_CLIENT1_PORT}",
            },
            command=f"python client.py",
        )
    print("Created smpc client 1 service.")

    # Start client 2
    print(f"\nWaiting for smpc client 2 to be ready...")
    try:
        docker_cli.containers.get(SMPC_CLIENT2_CONT_NAME)
    except docker.errors.NotFound:
        with open(
            path.join(TEST_ENV_CONFIG_FOLDER, LOCALWORKER2_SMPC_CONFIG_FILE)
        ) as fp:
            tmp = toml.load(fp)
            client_id = tmp["smpc"]["client_id"]
        docker_cli.containers.run(
            image=SMPC_CLUSTER_IMAGE,
            name=SMPC_CLIENT2_CONT_NAME,
            detach=True,
            ports={
                SMPC_CLIENT2_PORT: SMPC_CLIENT2_PORT,
            },
            environment={
                "PLAYER_REPO_0": f"http://{COMMON_IP}:{SMPC_PLAYER1_PORT2}",
                "PLAYER_REPO_1": f"http://{COMMON_IP}:{SMPC_PLAYER2_PORT2}",
                "PLAYER_REPO_2": f"http://{COMMON_IP}:{SMPC_PLAYER3_PORT2}",
                "COORDINATOR_URL": f"http://{COMMON_IP}:{SMPC_COORD_PORT}",
                "ID": client_id,
                "PORT": f"{SMPC_CLIENT2_PORT}",
            },
            command="python client.py",
        )
    print("Created smpc client 2 service.")

    yield

    # TODO Very slow development if containers are always removed afterwards
    # client1_cont = docker_cli.containers.get(SMPC_CLIENT1_CONT_NAME)
    # client1_cont.remove(v=True, force=True)
    # client2_cont = docker_cli.containers.get(SMPC_CLIENT2_CONT_NAME)
    # client2_cont.remove(v=True, force=True)


@pytest.fixture(scope="session")
def smpc_cluster(smpc_coordinator, smpc_players, smpc_clients):
    if USE_EXTERNAL_SMPC_CLUSTER:
        yield
        return

    print(f"\nWaiting for smpc cluster to be ready...")
    time.sleep(
        SMPC_CLUSTER_SLEEP_TIME
    )  # TODO Check when the smpc cluster is actually ready
    print(f"\nFinished waiting '{SMPC_CLUSTER_SLEEP_TIME}' secs for SMPC cluster.")
    yield


@pytest.fixture(scope="session")
def get_controller_testing_logger():
    return init_logger("TESTING", "DEBUG")

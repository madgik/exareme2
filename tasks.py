"""
Deployment script used for the development of the Exareme2.

In order to understand this script a basic knowledge of the system is required, this script
does not contain the documentation of the engine. The documentation of the celery,
in this script, is targeted to the specifics of the development deployment process.

This script deploys all the containers and api natively on your machine.
It deploys the containers on different ports and then configures the api to use the appropriate ports.

A worker service uses a configuration file either on the default location './exareme2/worker/config.toml'
or in the location of the env variable 'EXAREME2_WORKER_CONFIG_FILE', if the env variable is set.
This deployment script used for development, uses the env variable logic, therefore before deploying each
worker service the env variable is changed to the location of the worker api' config file.

In order for this script's celery to work the './configs/workers' folder should contain all the worker's config files
following the './exareme2/worker/config.toml' as template.
You can either create the files manually or using a '.deployment.toml' file with the following template
```
ip = "172.17.0.1"
log_level = "INFO"
framework_log_level ="INFO"
monetdb_image = "madgik/exareme2_db:dev1.3"

[controller]
port = 5000

[[workers]]
id = "globalworker"
monetdb_port=50000
rabbitmq_port=5670

[[workers]]
id = "localworker1"
monetdb_port=50001
rabbitmq_port=5671

[[workers]]
id = "localworker2"
monetdb_port=50002
rabbitmq_port=5672
```

and by running the command 'inv create-configs'.

The worker api are named after their config file. If a config file is named './configs/workers/localworker1.toml'
the worker service will be called 'localworker1' and should be referenced using that in the following celery.

Paths are subject to change so in the following documentation the global variables will be used.

"""

import copy
import csv
import glob
import itertools
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from contextlib import contextmanager
from enum import Enum
from itertools import cycle
from os import listdir
from os import path
from pathlib import Path
from textwrap import indent
from time import sleep

import requests
import toml
from invoke import UnexpectedExit
from invoke import task
from termcolor import colored

from exareme2.algorithms.exareme2.udfgen import udfio

PROJECT_ROOT = Path(__file__).parent
DEPLOYMENT_CONFIG_FILE = PROJECT_ROOT / ".deployment.toml"
WORKERS_CONFIG_DIR = PROJECT_ROOT / "configs" / "workers"
WORKER_CONFIG_TEMPLATE_FILE = PROJECT_ROOT / "exareme2" / "worker" / "config.toml"
CONTROLLER_CONFIG_DIR = PROJECT_ROOT / "configs" / "controller"
AGG_SERVER_CONFIG_DIR = PROJECT_ROOT / "configs" / "aggregation_server"
CONTROLLER_LOCALWORKERS_CONFIG_FILE = (
    PROJECT_ROOT / "configs" / "controller" / "localworkers_config.json"
)
CONTROLLER_CONFIG_TEMPLATE_FILE = (
    PROJECT_ROOT / "exareme2" / "controller" / "config.toml"
)
AGG_SERVER_DIR = PROJECT_ROOT / "aggregation_server"
AGG_SERVER_CONFIG_TEMPLATE_FILE = AGG_SERVER_DIR / "config.toml"
OUTDIR = Path("/tmp/exareme2/")
if not OUTDIR.exists():
    OUTDIR.mkdir()

CLEANUP_DIR = Path("/tmp/cleanup_entries/")
if not CLEANUP_DIR.exists():
    CLEANUP_DIR.mkdir()

TEST_DATA_FOLDER = PROJECT_ROOT / "tests" / "test_data"

EXAREME2_ALGORITHM_FOLDERS_ENV_VARIABLE = "EXAREME2_ALGORITHM_FOLDERS"
FLOWER_ALGORITHM_FOLDERS_ENV_VARIABLE = "FLOWER_ALGORITHM_FOLDERS"
EXAFLOW_ALGORITHM_FOLDERS_ENV_VARIABLE = "EXAFLOW_ALGORITHM_FOLDERS"
EXAREME2_WORKER_CONFIG_FILE = "EXAREME2_WORKER_CONFIG_FILE"
EXAREME2_CONTROLLER_CONFIG_FILE = "EXAREME2_CONTROLLER_CONFIG_FILE"
EXAREME2_AGG_SERVER_CONFIG_FILE = "EXAREME2_AGG_SERVER_CONFIG_FILE"
DATA_PATH = "DATA_PATH"

SMPC_COORDINATOR_PORT = 12314
SMPC_COORDINATOR_DB_PORT = 27017
SMPC_COORDINATOR_QUEUE_PORT = 6379
SMPC_PLAYER_BASE_PORT = 7000
SMPC_CLIENT_BASE_PORT = 9000
SMPC_COORDINATOR_NAME = "smpc_coordinator"
SMPC_COORDINATOR_DB_NAME = "smpc_coordinator_db"
SMPC_COORDINATOR_QUEUE_NAME = "smpc_coordinator_queue"
SMPC_PLAYER_BASE_NAME = "smpc_player"
SMPC_CLIENT_BASE_NAME = "smpc_client"


# TODO Add pre-celery when this is implemented https://github.com/pyinvoke/invoke/issues/170
# Right now if we call a task from another task, the "pre"-task is not executed


@task
def create_configs(c):
    """
    Create the worker and controller api config files, using 'DEPLOYMENT_CONFIG_FILE'.
    """
    if path.exists(WORKERS_CONFIG_DIR) and path.isdir(WORKERS_CONFIG_DIR):
        shutil.rmtree(WORKERS_CONFIG_DIR)
    WORKERS_CONFIG_DIR.mkdir(parents=True)

    if not Path(DEPLOYMENT_CONFIG_FILE).is_file():
        raise FileNotFoundError(
            f"Deployment config file '{DEPLOYMENT_CONFIG_FILE}' not found."
        )

    with open(DEPLOYMENT_CONFIG_FILE) as fp:
        deployment_config = toml.load(fp)

    with open(WORKER_CONFIG_TEMPLATE_FILE) as fp:
        template_worker_config = toml.load(fp)

    for worker in deployment_config["workers"]:
        worker_config = copy.deepcopy(template_worker_config)

        worker_config["identifier"] = worker["id"]
        worker_config["role"] = worker["role"]
        worker_config["federation"] = deployment_config["federation"]
        worker_config["log_level"] = deployment_config["log_level"]
        worker_config["framework_log_level"] = deployment_config["framework_log_level"]
        worker_config["controller"]["ip"] = deployment_config["ip"]
        worker_config["controller"]["port"] = deployment_config["controller"]["port"]
        worker_config["aggregation_server"]["enabled"] = deployment_config[
            "aggregation_server"
        ]["enabled"]
        worker_config["sqlite"]["db_name"] = worker["id"]
        worker_config["monetdb"]["enabled"] = deployment_config["monetdb"]["enabled"]
        if worker_config["monetdb"]["enabled"]:
            worker_config["monetdb"]["ip"] = deployment_config["ip"]
            worker_config["monetdb"]["port"] = worker["monetdb_port"]
            worker_config["monetdb"]["local_username"] = worker[
                "local_monetdb_username"
            ]
            worker_config["monetdb"]["local_password"] = worker[
                "local_monetdb_password"
            ]
            worker_config["monetdb"]["public_username"] = worker[
                "public_monetdb_username"
            ]
            worker_config["monetdb"]["public_password"] = worker[
                "public_monetdb_password"
            ]
            worker_config["monetdb"]["public_password"] = worker[
                "public_monetdb_password"
            ]

        worker_config["rabbitmq"]["ip"] = deployment_config["ip"]
        worker_config["rabbitmq"]["port"] = worker["rabbitmq_port"]

        worker_config["celery"]["tasks_timeout"] = deployment_config[
            "celery_tasks_timeout"
        ]
        worker_config["celery"]["run_udf_task_timeout"] = deployment_config[
            "celery_run_udf_task_timeout"
        ]

        worker_config["privacy"]["minimum_row_count"] = deployment_config["privacy"][
            "minimum_row_count"
        ]
        if worker["role"] == "GLOBALWORKER":
            worker_config["privacy"]["protect_local_data"] = False
        else:
            worker_config["privacy"]["protect_local_data"] = deployment_config[
                "privacy"
            ]["protect_local_data"]

        worker_config["smpc"]["enabled"] = deployment_config["smpc"]["enabled"]
        if worker_config["smpc"]["enabled"]:
            worker_config["smpc"]["optional"] = deployment_config["smpc"]["optional"]
            if coordinator_ip := deployment_config["smpc"].get("coordinator_ip"):
                if worker["role"] == "GLOBALWORKER":
                    worker_config["smpc"][
                        "coordinator_address"
                    ] = f"http://{coordinator_ip}:{SMPC_COORDINATOR_PORT}"
                else:
                    worker_config["smpc"]["client_id"] = worker["smpc_client_id"]
                    worker_config["smpc"][
                        "client_address"
                    ] = f"http://{coordinator_ip}:{worker['smpc_client_port']}"
            else:
                if worker["role"] == "GLOBALWORKER":
                    worker_config["smpc"][
                        "coordinator_address"
                    ] = f"http://{deployment_config['ip']}:{SMPC_COORDINATOR_PORT}"
                else:
                    worker_config["smpc"]["client_id"] = worker["id"]
                    worker_config["smpc"][
                        "client_address"
                    ] = f"http://{deployment_config['ip']}:{worker['smpc_client_port']}"

        worker_config_file = WORKERS_CONFIG_DIR / f"{worker['id']}.toml"
        with open(worker_config_file, "w+") as fp:
            toml.dump(worker_config, fp)

    # Create the controller config file
    with open(CONTROLLER_CONFIG_TEMPLATE_FILE) as fp:
        template_controller_config = toml.load(fp)
    controller_config = copy.deepcopy(template_controller_config)
    controller_config["node_identifier"] = "controller"
    controller_config["federation"] = deployment_config["federation"]
    controller_config["log_level"] = deployment_config["log_level"]
    controller_config["framework_log_level"] = deployment_config["framework_log_level"]

    controller_config["worker_landscape_aggregator_update_interval"] = (
        deployment_config["worker_landscape_aggregator_update_interval"]
    )
    controller_config["flower"]["enabled"] = deployment_config["flower"]["enabled"]
    controller_config["flower"]["execution_timeout"] = deployment_config["flower"][
        "execution_timeout"
    ]
    controller_config["flower"]["server_port"] = deployment_config["flower"][
        "server_port"
    ]
    controller_config["aggregation_server"]["enabled"] = deployment_config[
        "aggregation_server"
    ]["enabled"]
    controller_config["monetdb"]["enabled"] = deployment_config["monetdb"]["enabled"]
    controller_config["rabbitmq"]["celery_tasks_timeout"] = deployment_config[
        "celery_tasks_timeout"
    ]
    controller_config["rabbitmq"]["celery_cleanup_task_timeout"] = deployment_config[
        "celery_cleanup_task_timeout"
    ]
    controller_config["rabbitmq"]["celery_run_udf_task_timeout"] = deployment_config[
        "celery_run_udf_task_timeout"
    ]
    controller_config["deployment_type"] = "LOCAL"

    controller_config["localworkers"]["config_file"] = str(
        CONTROLLER_LOCALWORKERS_CONFIG_FILE
    )
    controller_config["localworkers"]["dns"] = ""
    controller_config["localworkers"]["port"] = ""

    controller_config["cleanup"]["contextids_cleanup_folder"] = str(CLEANUP_DIR)
    controller_config["cleanup"]["workers_cleanup_interval"] = deployment_config[
        "cleanup"
    ]["workers_cleanup_interval"]
    controller_config["cleanup"]["contextid_release_timelimit"] = deployment_config[
        "cleanup"
    ]["contextid_release_timelimit"]

    controller_config["smpc"]["enabled"] = deployment_config["smpc"]["enabled"]
    if controller_config["smpc"]["enabled"]:
        controller_config["smpc"]["optional"] = deployment_config["smpc"]["optional"]
        if coordinator_ip := deployment_config["smpc"].get("coordinator_ip"):
            controller_config["smpc"][
                "coordinator_address"
            ] = f"http://{coordinator_ip}:{SMPC_COORDINATOR_PORT}"
        else:
            controller_config["smpc"][
                "coordinator_address"
            ] = f"http://{deployment_config['ip']}:{SMPC_COORDINATOR_PORT}"

        controller_config["smpc"]["get_result_interval"] = deployment_config["smpc"][
            "get_result_interval"
        ]
        controller_config["smpc"]["get_result_max_retries"] = deployment_config["smpc"][
            "get_result_max_retries"
        ]

        controller_config["smpc"]["dp"]["enabled"] = deployment_config["smpc"]["dp"][
            "enabled"
        ]
        if controller_config["smpc"]["dp"]["enabled"]:
            controller_config["smpc"]["dp"]["sensitivity"] = deployment_config["smpc"][
                "dp"
            ]["sensitivity"]
            controller_config["smpc"]["dp"]["privacy_budget"] = deployment_config[
                "smpc"
            ]["dp"]["privacy_budget"]

    CONTROLLER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    controller_config_file = CONTROLLER_CONFIG_DIR / "controller.toml"
    with open(controller_config_file, "w+") as fp:
        toml.dump(controller_config, fp)

    # Create the controller localworkers config file
    localworkers_addresses = [
        f"{deployment_config['ip']}:{worker['rabbitmq_port']}"
        for worker in deployment_config["workers"]
    ]
    with open(CONTROLLER_LOCALWORKERS_CONFIG_FILE, "w+") as fp:
        json.dump(localworkers_addresses, fp)

    # Create the aggregation_server config file
    if not deployment_config["aggregation_server"]["enabled"]:
        return
    with open(AGG_SERVER_CONFIG_TEMPLATE_FILE) as fp:
        template_aggregation_server_config = toml.load(fp)
    aggregation_server_config = copy.deepcopy(template_aggregation_server_config)
    aggregation_server_config["enabled"] = deployment_config["aggregation_server"][
        "enabled"
    ]
    aggregation_server_config["port"] = deployment_config["aggregation_server"]["port"]
    aggregation_server_config["max_grpc_connections"] = deployment_config[
        "aggregation_server"
    ]["max_grpc_connections"]
    aggregation_server_config["max_wait_for_aggregation_inputs"] = deployment_config[
        "aggregation_server"
    ]["max_wait_for_aggregation_inputs"]
    aggregation_server_config["log_level"] = deployment_config["log_level"]

    AGG_SERVER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    aggregation_server_config_file = AGG_SERVER_CONFIG_DIR / "aggregation_server.toml"
    with open(aggregation_server_config_file, "w+") as fp:
        toml.dump(aggregation_server_config, fp)


@task
def install_dependencies(c):
    """Install project dependencies using poetry."""
    message("Installing dependencies...", Level.HEADER)
    cmd = "poetry install"
    run(c, cmd)


@task
def rm_containers(c, container_name=None, monetdb=False, rabbitmq=False, smpc=False):
    """
    Remove the specified docker containers, either by container or relative name.

    :param container_name: If set, removes the container with the specified name.
    :param monetdb: If True, it will remove all monetdb containers.
    :param rabbitmq: If True, it will remove all rabbitmq containers.
    :param smpc: If True, it will remove all smpc related containers.

    If nothing is set, nothing is removed.
    """
    names = []
    if monetdb:
        names.append("monetdb")
    if rabbitmq:
        names.append("rabbitmq")
    if smpc:
        names.append("smpc")
    if container_name:
        names.append(container_name)
    if not names:
        message(
            "You must specify at least one container family to remove (monetdb or/and rabbitmq)",
            level=Level.WARNING,
        )
    for name in names:
        container_ids = run(c, f"docker ps -qa --filter name={name}", show_ok=False)
        if container_ids.stdout:
            message(f"Removing {name} container(s)...", Level.HEADER)
            cmd = f"docker rm -vf $(docker ps -qa --filter name={name})"
            run(c, cmd)
        else:
            message(f"No {name} container to remove.", level=Level.HEADER)


@task(iterable=["worker"])
def create_monetdb(
    c, worker, image=None, log_level=None, nclients=None, monetdb_memory_limit=None
):
    """
    (Re)Create MonetDB container(s) for given worker(s). If the container exists, it will remove it and create it again.

    :param worker: A list of workers for which it will create the monetdb containers.
    :param image: The image to deploy. If not set, it will read it from the `DEPLOYMENT_CONFIG_FILE`.
    :param log_level: If not set, it will read it from the `DEPLOYMENT_CONFIG_FILE`.
    :param nclients: If not set, it will read it from the `DEPLOYMENT_CONFIG_FILE`.

    If an image is not provided it will use the 'monetdb_image' field from
    the 'DEPLOYMENT_CONFIG_FILE' ex. monetdb_image = "madgik/exareme2_db:dev1.2"

    The data of the monetdb container are not persisted. If the container is recreated, all data will be lost.
    """
    if not worker:
        message("Please specify a worker using --worker <worker>", Level.WARNING)
        sys.exit(1)

    if not image:
        image = get_deployment_config("monetdb_image")

    if not log_level:
        log_level = get_deployment_config("log_level")

    if not nclients:
        nclients = get_deployment_config("monetdb_nclients")

    if not monetdb_memory_limit:
        monetdb_memory_limit = get_deployment_config("monetdb_memory_limit")

    get_docker_image(c, image)

    udfio_full_path = path.abspath(udfio.__file__)

    worker_ids = worker
    for worker_id in worker_ids:
        container_name = f"monetdb-{worker_id}"
        rm_containers(c, container_name=container_name)

        worker_config_file = WORKERS_CONFIG_DIR / f"{worker_id}.toml"
        with open(worker_config_file) as fp:
            worker_config = toml.load(fp)
        monetdb_nclient_env_var = ""
        if worker_config["role"] == "GLOBALWORKER":
            monetdb_nclient_env_var = f"-e MONETDB_NCLIENTS={nclients}"
        container_ports = f"{worker_config['monetdb']['port']}:50000"
        message(
            f"Starting container {container_name} on ports {container_ports}...",
            Level.HEADER,
        )
        cmd = f"""docker run -d -P -p {container_ports} -e SOFT_RESTART_MEMORY_LIMIT={monetdb_memory_limit * 0.7} -e HARD_RESTART_MEMORY_LIMIT={monetdb_memory_limit * 0.85}  -e LOG_LEVEL={log_level} {monetdb_nclient_env_var} -e MAX_MEMORY={monetdb_memory_limit*1048576} {monetdb_nclient_env_var} -v {udfio_full_path}:/home/udflib/udfio.py -v {TEST_DATA_FOLDER}:{TEST_DATA_FOLDER} --name {container_name} --memory={monetdb_memory_limit}m {image}"""
        run(c, cmd)


@task(iterable=["worker"])
def init_system_tables(c, worker):
    """
    Initialize Sqlite with the system tables using mipdb.

    :param worker: A list of workers that will be initialized.
    """
    workers = worker
    for worker in workers:
        sqlite_path = f"{TEST_DATA_FOLDER}/{worker}.db"
        clean_sqlite(sqlite_path)
        message(
            f"Initializing system tables on sqlite with mipdb on worker: {worker}...",
            Level.HEADER,
        )
        cmd = f"""poetry run mipdb  {get_sqlite_path(worker)} init"""
        run(c, cmd)


@task
def update_wla(c):
    url = "http://localhost:5000/wla"
    try:
        response = requests.post(url, timeout=10)
    except requests.RequestException as exc:
        message(
            f"Warning: failed to update wla ({exc}); controller may still be starting.",
            Level.WARNING,
        )
        return

    if response.status_code != 200:
        raise Exception("Failed to update the wla")
    print("Successfully updated wla.")


@task(iterable=["port"])
def load_data(c, use_sockets=False, worker=None):
    """
    Load data into the specified DB from the 'TEST_DATA_FOLDER'.

    :param port: A list of ports, in which it will load the data. If not set, it will use the `WORKERS_CONFIG_DIR` files.
    :param use_sockets: Flag that determines if the data will be loaded via sockets or not.
    """

    def get_worker_configs():
        """
        Retrieve the configuration files of all workers.

        :return: A list of worker configurations.
        """
        config_files = [
            WORKERS_CONFIG_DIR / file for file in listdir(WORKERS_CONFIG_DIR)
        ]
        if not config_files:
            message(
                f"There are no worker config files to be used for data import. Folder: {WORKERS_CONFIG_DIR}",
                Level.WARNING,
            )
            sys.exit(1)

        worker_configs = []
        for worker_config_file in config_files:
            with open(worker_config_file) as fp:
                worker_config = toml.load(fp)
                worker_configs.append(worker_config)
        return worker_configs

    def filter_worker_configs(worker_configs, worker, node_type):
        """
        Filter worker configurations based on a specific worker identifier and node type.

        :param worker_configs: A list of all worker configurations.
        :param worker: The identifier of the worker to filter for.
        :param node_type: The type of node to filter for (default is "localworker").
        :return: A list of tuples containing worker identifiers and ports.
        """
        return [
            (config["identifier"], config["monetdb"]["port"])
            for config in worker_configs
            if (not worker or config["identifier"] == worker)
            and config["role"] == node_type
        ]

    def read_data_model_metadata(cdes_file):
        """Read the data model metadata definition."""

        with open(cdes_file) as data_model_metadata_file:
            data_model_metadata = json.load(data_model_metadata_file)

        data_model_code = data_model_metadata["code"]
        data_model_version = data_model_metadata["version"]

        return data_model_code, data_model_version

    def calculate_dataset_distribution(dirpath, filenames, worker_id_and_ports):
        """Map datasets to workers following the existing allocation rules."""
        dataset_distribution = {worker_id: [] for worker_id, _ in worker_id_and_ports}

        if not worker_id_and_ports:
            return dataset_distribution

        if len(worker_id_and_ports) == 1:
            worker_id = worker_id_and_ports[0][0]
            dataset_distribution[worker_id] = sorted(
                [
                    os.path.join(dirpath, file)
                    for file in filenames
                    if file.endswith(".csv") and not file.endswith("test.csv")
                ]
            )
            return dataset_distribution

        first_worker_id = worker_id_and_ports[0][0]
        first_worker_csvs = sorted(
            [
                os.path.join(dirpath, file)
                for file in filenames
                if file.endswith("0.csv") and not file.endswith("10.csv")
            ]
        )
        dataset_distribution[first_worker_id].extend(first_worker_csvs)

        remaining_csvs = sorted(
            [
                os.path.join(dirpath, file)
                for file in filenames
                if file.endswith(".csv")
                and not file.endswith("0.csv")
                and not file.endswith("test.csv")
            ]
        )
        worker_cycle = itertools.cycle(worker_id_and_ports[1:])
        for csv_path in remaining_csvs:
            worker_id, _ = next(worker_cycle)
            dataset_distribution[worker_id].append(csv_path)

        return dataset_distribution

    def calculate_test_dataset_distribution(dirpath, filenames, worker_id_and_ports):
        """
        Calculate the distribution for datasets ending with 'test'.

        :param dirpath: Directory path of the current dataset.
        :param filenames: List of filenames in the current directory.
        :param worker_id_and_ports: A list of tuples containing worker identifiers and ports.
        """
        dataset_distribution = {worker_id: [] for worker_id, _ in worker_id_and_ports}
        test_csvs = sorted(
            [
                os.path.join(dirpath, file)
                for file in filenames
                if file.endswith("test.csv")
            ]
        )

        if not worker_id_and_ports:
            return dataset_distribution

        worker_id, _ = worker_id_and_ports[0]
        for csv_path in test_csvs:
            dataset_distribution[worker_id].append(csv_path)

        return dataset_distribution

    def concatenate_csv_files(csv_paths, destination_path):
        """Concatenate CSV files using the canonical header order."""
        if not csv_paths:
            return

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        csv_paths = sorted(csv_paths)

        canonical_header = None

        with open(destination_path, "w", newline="") as destination_file:
            writer = None
            for csv_path in csv_paths:
                with open(csv_path, newline="") as source_file:
                    reader = csv.DictReader(source_file)
                    if reader.fieldnames is None:
                        continue

                    if canonical_header is None:
                        canonical_header = reader.fieldnames
                        writer = csv.DictWriter(
                            destination_file, fieldnames=canonical_header
                        )
                        writer.writeheader()
                    else:
                        if set(reader.fieldnames) != set(canonical_header):
                            message(
                                "Inconsistent headers detected while concatenating datasets.",
                                Level.ERROR,
                            )
                            raise ValueError(
                                "Dataset headers do not match the canonical data model header."
                            )

                    for row in reader:
                        writer.writerow(
                            {field: row.get(field, "") for field in canonical_header}
                        )

    def create_combined_datasets(worker_dataset_structure, worker_ids):
        """Create combined dataset folders per worker and data model."""

        combined_root = TEST_DATA_FOLDER / "__combined"
        if combined_root.exists():
            shutil.rmtree(combined_root)
        combined_root.mkdir(exist_ok=True)

        combined_folder_structure = defaultdict(dict)
        worker_directories = {}

        for worker_id in worker_ids:
            worker_dir = combined_root / worker_id
            worker_dir.mkdir(exist_ok=True)
            worker_directories[worker_id] = worker_dir

            data_models = worker_dataset_structure.get(worker_id, {})

            for data_model_key in sorted(data_models.keys()):
                data_model_entry = data_models[data_model_key]
                dataset_paths = sorted(set(data_model_entry.get("datasets", [])))
                metadata_path = data_model_entry.get("metadata")

                if not dataset_paths or not metadata_path:
                    continue

                if not os.path.exists(metadata_path):
                    message(
                        f"Skipping data model '{data_model_key[0]}:{data_model_key[1]}' for worker {worker_id} (metadata missing).",
                        Level.WARNING,
                    )
                    continue

                data_model_code, data_model_version = data_model_key
                data_model_dir = worker_dir / f"{data_model_code}_{data_model_version}"
                data_model_dir.mkdir(exist_ok=True)

                metadata_destination = data_model_dir / "CDEsMetadata.json"
                shutil.copyfile(metadata_path, metadata_destination)

                combined_csv_path = data_model_dir / f"{data_model_code}.csv"
                concatenate_csv_files(dataset_paths, combined_csv_path)

                combined_folder_structure[worker_id][data_model_key] = {
                    "folder": str(data_model_dir),
                    "metadata": str(metadata_destination),
                    "combined_csv": str(combined_csv_path),
                    "source_datasets": [str(path) for path in dataset_paths],
                }

        return combined_root, worker_directories, combined_folder_structure

    def load_combined_folders(worker_dirs, port_lookup, use_sockets):
        """Load combined folders into MonetDB for each worker in parallel."""

        load_jobs = []
        for worker_id, port in port_lookup.items():
            worker_dir = worker_dirs.get(worker_id)
            if not worker_dir or not worker_dir.exists():
                continue

            if not any(worker_dir.iterdir()):
                message(
                    f"No combined datasets found for worker: {worker_id}. Skipping load-folder command.",
                    Level.WARNING,
                )
                continue

            load_jobs.append((worker_id, port, worker_dir))

        if not load_jobs:
            return

        max_workers = min(len(load_jobs), max(os.cpu_count() or 1, 1))

        def execute_load_folder(worker_id, port, worker_dir):
            message(
                f"Loading combined datasets for worker: {worker_id} from {worker_dir}...",
                Level.HEADER,
            )
            cmd = (
                f"poetry run mipdb {get_monetdb_configs_in_mipdb_format(port)} "
                f"{get_sqlite_path(worker_id)} load-folder {worker_dir}"
            )
            if get_deployment_config("monetdb", subconfig="enabled") and use_sockets:
                cmd += " --no-copy"

            completed_process = subprocess.run(
                cmd,
                shell=True,
                check=False,
                text=True,
            )
            if completed_process.returncode != 0:
                message(
                    f"Failed to load combined datasets for worker: {worker_id} (exit code {completed_process.returncode}).",
                    Level.ERROR,
                )
                raise UnexpectedExit(
                    f"load-folder failed for worker {worker_id} with exit code {completed_process.returncode}"
                )

        errors = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(execute_load_folder, worker_id, port, worker_dir)
                for worker_id, port, worker_dir in load_jobs
            ]

            for future in futures:
                try:
                    future.result()
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)

        if errors:
            raise RuntimeError(
                "One or more load-folder operations failed. Check the logs above for details."
            )

    worker_dataset_structure = defaultdict(dict)

    # Retrieve and filter worker configurations for local workers
    worker_configs = get_worker_configs()
    local_worker_id_and_ports = filter_worker_configs(
        worker_configs, worker, "LOCALWORKER"
    )

    if not local_worker_id_and_ports:
        raise Exception("Local worker config files cannot be loaded.")

    all_worker_ids = {worker_id for worker_id, _ in local_worker_id_and_ports}
    worker_port_lookup = {
        worker_id: port for worker_id, port in local_worker_id_and_ports
    }

    # Process each dataset in the TEST_DATA_FOLDER for local workers
    for dirpath, dirnames, filenames in os.walk(TEST_DATA_FOLDER):
        if "__combined" in dirnames:
            dirnames.remove("__combined")
        if "CDEsMetadata.json" not in filenames:
            continue
        cdes_file = os.path.join(dirpath, "CDEsMetadata.json")

        data_model_code, data_model_version = read_data_model_metadata(cdes_file)
        data_model_key = (data_model_code, data_model_version)

        dataset_distribution = calculate_dataset_distribution(
            dirpath, filenames, local_worker_id_and_ports
        )

        for worker_id, csv_paths in dataset_distribution.items():
            if not csv_paths:
                continue

            entry = worker_dataset_structure[worker_id].get(data_model_key)
            if not entry:
                entry = {"metadata": cdes_file, "datasets": []}
                worker_dataset_structure[worker_id][data_model_key] = entry

            if not entry.get("metadata"):
                entry["metadata"] = cdes_file
            entry["datasets"].extend(csv_paths)

    # Retrieve and filter worker configurations for global worker
    global_worker_id_and_ports = filter_worker_configs(
        worker_configs, worker, "GLOBALWORKER"
    )

    if not global_worker_id_and_ports:
        raise Exception("Global worker config files cannot be loaded.")

    all_worker_ids.update(worker_id for worker_id, _ in global_worker_id_and_ports)
    worker_port_lookup.update(
        {worker_id: port for worker_id, port in global_worker_id_and_ports}
    )

    # Process each dataset in the TEST_DATA_FOLDER for global worker
    for dirpath, dirnames, filenames in os.walk(TEST_DATA_FOLDER):
        if "__combined" in dirnames:
            dirnames.remove("__combined")
        if "CDEsMetadata.json" not in filenames:
            continue
        cdes_file = os.path.join(dirpath, "CDEsMetadata.json")

        data_model_code, data_model_version = read_data_model_metadata(cdes_file)
        data_model_key = (data_model_code, data_model_version)

        test_dataset_distribution = calculate_test_dataset_distribution(
            dirpath,
            filenames,
            global_worker_id_and_ports,
        )

        for worker_id, csv_paths in test_dataset_distribution.items():
            if not csv_paths:
                continue

            entry = worker_dataset_structure[worker_id].get(data_model_key)
            if not entry:
                entry = {"metadata": cdes_file, "datasets": []}
                worker_dataset_structure[worker_id][data_model_key] = entry

            if not entry.get("metadata"):
                entry["metadata"] = cdes_file
            entry["datasets"].extend(csv_paths)

    if all_worker_ids:
        (
            combined_root,
            worker_directories,
            combined_folder_structure,
        ) = create_combined_datasets(worker_dataset_structure, sorted(all_worker_ids))

        message(
            f"Combined dataset folders prepared at: {combined_root}",
            Level.BODY,
        )

        for worker_id, data_models in combined_folder_structure.items():
            message(
                f"  Worker {worker_id} -> {len(data_models)} data model(s) combined",
                Level.BODY,
            )

        load_combined_folders(worker_directories, worker_port_lookup, use_sockets)


def get_sqlite_path(worker_id):
    return f"--sqlite {TEST_DATA_FOLDER}/{worker_id}.db"


def get_monetdb_configs_in_mipdb_format(port):
    if not get_deployment_config("monetdb", subconfig="enabled"):
        return "--no-monetdb "
    return (
        f"--monetdb "
        f"--ip 127.0.0.1 "
        f"--port {port} "
        f"--username admin "
        f"--password executor "
        f"--db-name db"
    )


@task(iterable=["worker"])
def create_rabbitmq(c, worker, rabbitmq_image=None):
    """
    (Re)Create RabbitMQ container(s) of given worker(s). If the container exists, remove it and create it again.

    :param worker: A list of workers for which to (re)create the rabbitmq containers.
    :param rabbitmq_image: The image to deploy. If not set, it will read it from the `DEPLOYMENT_CONFIG_FILE`.

    """
    if not worker:
        message("Please specify a worker using --worker <worker>", Level.WARNING)
        sys.exit(1)

    if not rabbitmq_image:
        rabbitmq_image = get_deployment_config("rabbitmq_image")

    get_docker_image(c, rabbitmq_image)

    worker_ids = worker
    for worker_id in worker_ids:
        container_name = f"rabbitmq-{worker_id}"
        rm_containers(c, container_name=container_name)

        worker_config_file = WORKERS_CONFIG_DIR / f"{worker_id}.toml"
        with open(worker_config_file) as fp:
            worker_config = toml.load(fp)
        queue_port = f"{worker_config['rabbitmq']['port']}:5672"
        api_port = f"{worker_config['rabbitmq']['port']+10000}:15672"
        message(
            f"Starting container {container_name} on ports {queue_port}...",
            Level.HEADER,
        )
        cmd = f"docker run -d -p {queue_port} -p {api_port} --name {container_name} {rabbitmq_image}"
        run(c, cmd)


@task(iterable=["worker"])
def verify_rabbitmq(c, worker):
    """Ensure RabbitMQ containers are healthy for the provided worker ids."""

    if not worker:
        message("Please specify a worker using --worker <worker>", Level.WARNING)
        sys.exit(1)

    worker_ids = worker

    for worker_id in worker_ids:
        container_name = f"rabbitmq-{worker_id}"

        cmd = f"docker inspect --format='{{{{json .State.Health}}}}' {container_name}"
        # Wait until rabbitmq is healthy
        message(
            f"Waiting for container {container_name} to become healthy...",
            Level.HEADER,
        )
        for _ in range(100):
            status = run(c, cmd, raise_error=True, wait=True, show_ok=False)

            if '"Status":"healthy"' not in status.stdout:
                spin_wheel(time=2)
            else:
                message("Ok", Level.SUCCESS)
                break
        else:
            message("Cannot configure RabbitMQ", Level.ERROR)
            sys.exit(1)


@task
def kill_worker(c, worker=None, all_=False):
    """
    Kill the worker(s) service(s).

    :param worker: The worker service to kill.
    :param all_: If set, all worker api will be killed.
    """

    if all_:
        worker_pattern = ""
    elif worker:
        worker_pattern = worker
    else:
        message(
            "Please specify a worker using --worker <worker> or use --all",
            Level.WARNING,
        )
        sys.exit(1)

    res_bin = run(
        c,
        f"ps aux | grep '[c]elery' | grep 'worker' | grep '{worker_pattern}' ",
        warn=True,
        show_ok=False,
    )

    if res_bin.ok:
        message(
            f"Killing previous celery instance(s) with pattern '{worker_pattern}' ...",
            Level.HEADER,
        )

        # We need to kill the celery worker processes with the "worker_pattern", if provided.
        # First we kill the parent process (celery workers' parent) if there is one, when "worker_pattern is provided,
        # and then we kill all the celery worker processes with/without a pattern.
        cmd = (
            f"pid=$(ps aux | grep '[c]elery' | grep 'worker' | grep '{worker_pattern}' | awk '{{print $2}}') "
            f"&& pgrep -P $pid | xargs kill -9 "
        )
        run(c, cmd, warn=True, show_ok=False)
        cmd = (
            f"pid=$(ps aux | grep '[c]elery' | grep 'worker' | grep '{worker_pattern}' | awk '{{print $2}}') "
            f"&& kill -9 $pid "
        )
        run(c, cmd, warn=True)
    else:
        message("No celery instances found", Level.HEADER)


def validate_algorithm_folders(folders, name):
    """Validates and retrieves the algorithm folder configuration."""
    if not folders:
        folders = get_deployment_config(name)
    if not isinstance(folders, str):
        raise ValueError(f"The {name} configuration must be a comma-separated string.")
    return folders


@contextmanager
def env_prefixes(c, env_vars):
    """
    A context manager that applies a set of environment variable prefixes
    using an ExitStack.
    """
    with ExitStack() as stack:
        for key, value in env_vars.items():
            stack.enter_context(c.prefix(f"export {key}={value}"))
        yield


@task
def start_worker(
    c,
    worker=None,
    all_=False,
    framework_log_level=None,
    detached=False,
    exareme2_algorithm_folders=None,
    flower_algorithm_folders=None,
    exaflow_algorithm_folders=None,
):
    """
    (Re)Start the worker(s) service(s). If a worker service is running, stop and start it again.

    :param worker: The worker to start, using the proper file in the `WORKERS_CONFIG_DIR`.
    :param all_: If set, the workers of which the configuration file exists, will be started.
    :param framework_log_level: If not provided, it will look into the `DEPLOYMENT_CONFIG_FILE`.
    :param detached: If set to True, it will start the service in the background.
    :param exareme2_algorithm_folders: Used from the api. If not provided, it looks in the `DEPLOYMENT_CONFIG_FILE`.
    :param flower_algorithm_folders: Used from the api. If not provided, it looks in the `DEPLOYMENT_CONFIG_FILE`.

    The containers related to the api remain unchanged.
    """

    if not framework_log_level:
        framework_log_level = get_deployment_config("framework_log_level")

    # Validate algorithm folders
    exareme2_algorithm_folders = validate_algorithm_folders(
        exareme2_algorithm_folders, "exareme2_algorithm_folders"
    )
    flower_algorithm_folders = validate_algorithm_folders(
        flower_algorithm_folders, "flower_algorithm_folders"
    )
    exaflow_algorithm_folders = validate_algorithm_folders(
        exaflow_algorithm_folders, "exaflow_algorithm_folders"
    )

    # Retrieve and sort worker ids to avoid accidental removal of similarly named ids
    worker_ids = sorted(get_worker_ids(all_, worker))

    for worker_id in worker_ids:
        kill_worker(c, worker_id)
        message(f"Starting Worker {worker_id}...", Level.HEADER)
        worker_config_file = WORKERS_CONFIG_DIR / f"{worker_id}.toml"

        # Build environment variables dictionary
        env_vars = {
            EXAREME2_ALGORITHM_FOLDERS_ENV_VARIABLE: exareme2_algorithm_folders,
            FLOWER_ALGORITHM_FOLDERS_ENV_VARIABLE: flower_algorithm_folders,
            EXAFLOW_ALGORITHM_FOLDERS_ENV_VARIABLE: exaflow_algorithm_folders,
            EXAREME2_WORKER_CONFIG_FILE: worker_config_file,
            DATA_PATH: TEST_DATA_FOLDER.as_posix(),
        }

        # Use the helper context manager to apply environment variable prefixes
        with env_prefixes(c, env_vars):
            outpath = OUTDIR / f"{worker_id}.out"

            # Build and run the command based on the detached/all_ flags
            if detached or all_:
                cmd = (
                    f"PYTHONPATH={PROJECT_ROOT}: poetry run celery "
                    f"-A exareme2.worker.utils.celery_app worker -l {framework_log_level} > {outpath} "
                    f"--pool=eventlet --purge 2>&1"
                )
                run(c, cmd, wait=False)
            else:
                cmd = (
                    f"PYTHONPATH={PROJECT_ROOT} poetry run celery -A "
                    f"exareme2.worker.utils.celery_app worker -l {framework_log_level} --pool=eventlet --purge"
                )
                run(c, cmd, attach_=True)


@task
def kill_aggregation_server(c):
    """
    Kill any running aggregation_server.server processes.
    """
    res = c.run("ps aux | grep '[a]ggregation_server.server'", warn=True, hide="both")
    if res.ok and res.stdout.strip():
        message("Killing existing aggregation_server ...", Level.HEADER)
        c.run(
            "ps aux | grep '[a]ggregation_server.server' "
            "| awk '{print $2}' | xargs kill -9",
            warn=True,
        )
        message("Ok", Level.SUCCESS)
    else:
        message("No aggregation_server process found.", Level.HEADER)


@task(pre=[kill_aggregation_server])
def start_aggregation_server(c, detached: bool = False):
    """
    Start the aggregation_server gRPC service.
    If detached=True, run in background and log to OUTDIR/aggregation_server.out.
    """
    message("Starting aggregation_server…", Level.HEADER)

    if not AGG_SERVER_CONFIG_TEMPLATE_FILE.exists():
        message(f"Config not found: {AGG_SERVER_CONFIG_TEMPLATE_FILE}", Level.ERROR)
        return

    # Build environment for the server process
    env = os.environ.copy()
    env["AGG_SERVER_CONFIG_FILE"] = str(AGG_SERVER_CONFIG_TEMPLATE_FILE)

    # cd into the aggregation_server folder so Poetry picks up its own pyproject.toml
    run_cmd = (
        f"cd {AGG_SERVER_DIR!s} && " "poetry run python -m aggregation_server.server"
    )

    if detached:
        logf = OUTDIR / "aggregation_server.out"
        c.run(
            f"{run_cmd} >> {logf!s} 2>&1 &",
            env=env,
            pty=False,
            disown=True,
            warn=True,
        )
        message("Ok.", Level.SUCCESS)
    else:
        c.run(
            run_cmd,
            env=env,
            pty=True,
        )


@task
def kill_controller(c):
    """Kill the controller service."""
    HYPERCORN_PROCESS_NAME = "[f]rom multiprocessing.spawn import spawn_main;"
    res = run(c, f"ps aux | grep '{HYPERCORN_PROCESS_NAME}'", warn=True, show_ok=False)
    if res.ok:
        message("Killing previous Hypercorn instances...", Level.HEADER)
        cmd = f"ps aux | grep '{HYPERCORN_PROCESS_NAME}' | awk '{{ print $2}}' | xargs kill -9 && sleep 5"
        run(c, cmd)
    else:
        message("No hypercorn instance found", Level.HEADER)


@task
def start_controller(
    c,
    detached=False,
    exareme2_algorithm_folders=None,
    flower_algorithm_folders=None,
    exaflow_algorithm_folders=None,
):
    """
    (Re)Start the controller service. If the service is already running, stop and start it again.
    """

    # Validate algorithm folders
    exareme2_algorithm_folders = validate_algorithm_folders(
        exareme2_algorithm_folders, "exareme2_algorithm_folders"
    )
    flower_algorithm_folders = validate_algorithm_folders(
        flower_algorithm_folders, "flower_algorithm_folders"
    )
    exaflow_algorithm_folders = validate_algorithm_folders(
        exaflow_algorithm_folders, "exaflow_algorithm_folders"
    )

    kill_controller(c)
    message("Starting Controller...", Level.HEADER)
    controller_config_file = CONTROLLER_CONFIG_DIR / "controller.toml"

    # Build a dictionary of environment variables for the controller
    env_vars = {
        EXAREME2_ALGORITHM_FOLDERS_ENV_VARIABLE: exareme2_algorithm_folders,
        FLOWER_ALGORITHM_FOLDERS_ENV_VARIABLE: flower_algorithm_folders,
        EXAFLOW_ALGORITHM_FOLDERS_ENV_VARIABLE: exaflow_algorithm_folders,
        EXAREME2_CONTROLLER_CONFIG_FILE: controller_config_file,
    }

    # Use the helper context manager to apply environment variable prefixes
    with env_prefixes(c, env_vars):
        outpath = OUTDIR / "controller.out"
        if detached:
            cmd = (
                f"PYTHONPATH={PROJECT_ROOT} poetry run hypercorn --config python:exareme2.controller.quart.hypercorn_config "
                f"-b 0.0.0.0:5000 exareme2.controller.quart.app:app >> {outpath} 2>&1"
            )
            run(c, cmd, wait=False)
        else:
            cmd = (
                f"PYTHONPATH={PROJECT_ROOT} poetry run hypercorn --config python:exareme2.controller.quart.hypercorn_config "
                f"-b 0.0.0.0:5000 exareme2.controller.quart.app:app"
            )
            run(c, cmd, attach_=True)


@task
def deploy(
    c,
    install_dep=True,
    log_level=None,
    framework_log_level=None,
    monetdb_image=None,
    monetdb_nclients=None,
    exareme2_algorithm_folders=None,
    flower_algorithm_folders=None,
    exaflow_algorithm_folders=None,
    smpc=None,
):
    """
    Install dependencies, (re)create all the containers and (re)start all the api.

    :param install_dep: Install dependencies or not.
    :param log_level: Used for the dev logs. If not provided, it looks in the `DEPLOYMENT_CONFIG_FILE`.
    :param framework_log_level: Used for the engine api. If not provided, it looks in the `DEPLOYMENT_CONFIG_FILE`.
    :param monetdb_image: Used for the db containers. If not provided, it looks in the `DEPLOYMENT_CONFIG_FILE`.
    :param monetdb_nclients: Used for the db containers. If not provided, it looks in the `DEPLOYMENT_CONFIG_FILE`.
    :param exareme2_algorithm_folders: Used from the api. If not provided, it looks in the `DEPLOYMENT_CONFIG_FILE`.
    :param flower_algorithm_folders: Used from the api. If not provided, it looks in the `DEPLOYMENT_CONFIG_FILE`.
    :param exaflow_algorithm_folders: Used from the api. If not provided, it looks in the `DEPLOYMENT_CONFIG_FILE`.
    :param smpc: Deploy the SMPC cluster as well. If not provided, it looks in the `DEPLOYMENT_CONFIG_FILE`.
    """

    if not log_level:
        log_level = get_deployment_config("log_level")

    if not framework_log_level:
        framework_log_level = get_deployment_config("framework_log_level")

    start_monetdb = get_deployment_config("monetdb", subconfig="enabled")
    if not monetdb_image:
        monetdb_image = get_deployment_config("monetdb_image")

    if not monetdb_nclients:
        monetdb_nclients = get_deployment_config("monetdb_nclients")

    if not exareme2_algorithm_folders:
        exareme2_algorithm_folders = get_deployment_config("exareme2_algorithm_folders")

    if not flower_algorithm_folders:
        flower_algorithm_folders = get_deployment_config("flower_algorithm_folders")

    if not exaflow_algorithm_folders:
        exaflow_algorithm_folders = get_deployment_config("exaflow_algorithm_folders")

    start_aggregation_server_ = get_deployment_config(
        "aggregation_server", subconfig="enabled"
    )

    if smpc is None:
        smpc = get_deployment_config("smpc", subconfig="enabled")

    if install_dep:
        install_dependencies(c)

    # Start WORKER api
    config_files = [WORKERS_CONFIG_DIR / file for file in listdir(WORKERS_CONFIG_DIR)]
    if not config_files:
        message(
            f"There are no worker config files to be used for deployment. Folder: {WORKERS_CONFIG_DIR}",
            Level.WARNING,
        )
        sys.exit(1)

    worker_ids = []
    for worker_config_file in config_files:
        with open(worker_config_file) as fp:
            worker_config = toml.load(fp)
        worker_ids.append(worker_config["identifier"])

    create_rabbitmq(c, worker=worker_ids)

    worker_ids.sort()  # Sorting the ids protects removing a similarly named id, localworker1 would remove localworker10.
    if start_monetdb:
        create_monetdb(
            c,
            worker=worker_ids,
            image=monetdb_image,
            log_level=log_level,
            nclients=monetdb_nclients,
        )
    init_system_tables(c, worker=worker_ids)

    if start_aggregation_server_:
        start_aggregation_server(c, detached=True)

    verify_rabbitmq(c, worker=worker_ids)

    start_worker(
        c,
        all_=True,
        framework_log_level=framework_log_level,
        detached=True,
        exareme2_algorithm_folders=exareme2_algorithm_folders,
        flower_algorithm_folders=flower_algorithm_folders,
        exaflow_algorithm_folders=exaflow_algorithm_folders,
    )

    # Start CONTROLLER service
    start_controller(
        c,
        detached=True,
        exareme2_algorithm_folders=exareme2_algorithm_folders,
        flower_algorithm_folders=flower_algorithm_folders,
        exaflow_algorithm_folders=exaflow_algorithm_folders,
    )

    if smpc and not get_deployment_config("smpc", subconfig="coordinator_ip"):
        deploy_smpc(c)


@task
def attach(c, worker=None, controller=False, db=None):
    """
    Attach to a worker/controller service or a db container.

    :param worker: The worker service name to which to attach.
    :param controller: Attach to controller flag.
    :param db: The db container name to which to attach.
    """
    if (worker or controller) and not (worker and controller):
        fname = worker or "controller"
        outpath = OUTDIR / (fname + ".out")
        cmd = f"tail -f {outpath}"
        run(c, cmd, attach_=True)
    elif db:
        run(c, f"docker exec -it {db} mclient db", attach_=True)
    else:
        message("You must attach to Worker, Controller or DB", Level.WARNING)
        sys.exit(1)


@task
def cleanup(c):
    """Kill all worker/controller api and remove all monetdb/rabbitmq containers."""
    kill_controller(c)
    kill_aggregation_server(c)
    kill_worker(c, all_=True)
    rm_containers(c, monetdb=True, rabbitmq=True, smpc=True)

    # Create a pattern for .db files
    pattern = os.path.join(TEST_DATA_FOLDER, "*.db")

    # Delete each .db file
    for sqlite_path in glob.glob(pattern):
        clean_sqlite(sqlite_path)

    combined_root_path = TEST_DATA_FOLDER / "__combined"
    if combined_root_path.exists():
        try:
            message(f"Removing {combined_root_path}...", level=Level.HEADER)

            shutil.rmtree(combined_root_path)
        except OSError:
            message(
                f"Failed to remove combined dataset directory {combined_root_path}",
                Level.WARNING,
            )

    if OUTDIR.exists():
        message(f"Removing {OUTDIR}...", level=Level.HEADER)
        for outpath in OUTDIR.glob("*.out"):
            outpath.unlink()
        OUTDIR.rmdir()
        message("Ok", level=Level.SUCCESS)
    if CLEANUP_DIR.exists():
        message(f"Removing {CLEANUP_DIR}...", level=Level.HEADER)
        for cleanup_file in CLEANUP_DIR.glob("*.toml"):
            cleanup_file.unlink()
        CLEANUP_DIR.rmdir()
        message("Ok", level=Level.SUCCESS)


def clean_sqlite(sqlite_path):
    try:
        if os.path.exists(sqlite_path):
            message(f"Removing {sqlite_path}...", level=Level.HEADER)
            os.remove(sqlite_path)
            message("Ok", level=Level.SUCCESS)
    except Exception as e:
        print(f"Error deleting {sqlite_path}: {e}")


@task
def start_flower(c, worker=None, all_=False):
    """
    (Re)Start flower monitoring tool. If flower is already running, stop ir and start it again.

    :param worker: The worker service, for which to create the flower monitoring.
    :param all_: If set, it will create monitoring for all worker api in the `WORKERS_CONFIG_DIR`.
    """

    kill_all_flowers(c)

    FLOWER_PORT = 5550

    worker_ids = get_worker_ids(all_, worker)
    worker_ids.sort()

    for worker_id in worker_ids:
        worker_config_file = WORKERS_CONFIG_DIR / f"{worker_id}.toml"
        with open(worker_config_file) as fp:
            worker_config = toml.load(fp)

        ip = worker_config["rabbitmq"]["ip"]
        port = worker_config["rabbitmq"]["port"]
        api_port = port + 10000
        user_and_password = (
            worker_config["rabbitmq"]["user"]
            + ":"
            + worker_config["rabbitmq"]["password"]
        )
        vhost = worker_config["rabbitmq"]["vhost"]
        flower_url = ip + ":" + str(port)
        broker = f"amqp://{user_and_password}@{flower_url}/{vhost}"
        broker_api = f"http://{user_and_password}@{ip + ':' + str(api_port)}/api/"

        flower_index = worker_ids.index(worker_id)
        flower_port = FLOWER_PORT + flower_index

        message(f"Starting flower container for worker {worker_id}...", Level.HEADER)
        command = f"docker run --name flower-{worker_id} -d -p {flower_port}:5555 mher/flower:0.9.5 flower --broker={broker} --broker-api={broker_api}"
        run(c, command)
        cmd = "docker ps | grep '[f]lower'"
        run(c, cmd, warn=True, show_ok=False)
        message(f"Visit me at http://localhost:{flower_port}", Level.HEADER)


@task
def kill_all_flowers(c):
    """Kill all flower instances."""
    container_ids = run(c, "docker ps -qa --filter name=flower", show_ok=False)
    if container_ids.stdout:
        message("Killing Flower instances and removing containers...", Level.HEADER)
        cmd = f"docker container kill flower & docker rm -vf $(docker ps -qa --filter name=flower)"
        run(c, cmd)
        message("Flower has withered away", Level.HEADER)
    else:
        message(f"No flower container to remove", level=Level.HEADER)


def start_smpc_coordinator_db(c, image):
    container_ports = f"{SMPC_COORDINATOR_DB_PORT}:27017"
    message(
        f"Starting container {SMPC_COORDINATOR_DB_NAME} on ports {container_ports}...",
        Level.HEADER,
    )
    env_variables = (
        "-e MONGO_INITDB_ROOT_USERNAME=sysadmin "
        "-e MONGO_INITDB_ROOT_PASSWORD=123qwe "
    )
    cmd = f"docker run -d -p {container_ports} {env_variables} --name {SMPC_COORDINATOR_DB_NAME} {image}"
    run(c, cmd)


def start_smpc_coordinator_queue(c, image):
    container_ports = f"{SMPC_COORDINATOR_QUEUE_PORT}:6379"
    message(
        f"Starting container {SMPC_COORDINATOR_QUEUE_NAME} on ports {container_ports}...",
        Level.HEADER,
    )
    container_cmd = "redis-server --requirepass agora"
    cmd = f"""docker run -d -p {container_ports} -e REDIS_REPLICATION_MODE=master --name {SMPC_COORDINATOR_QUEUE_NAME} {image} {container_cmd}"""
    run(c, cmd)


def start_smpc_coordinator_container(c, ip, image):
    container_ports = f"{SMPC_COORDINATOR_PORT}:12314"
    message(
        f"Starting container {SMPC_COORDINATOR_NAME} on ports {container_ports}...",
        Level.HEADER,
    )
    container_cmd = "python coordinator.py"
    env_variables = (
        f"-e PLAYER_REPO_0=http://{ip}:7000 "
        f"-e PLAYER_REPO_1=http://{ip}:7001 "
        f"-e PLAYER_REPO_2=http://{ip}:7002 "
        f"-e REDIS_HOST={ip} "
        f"-e REDIS_PORT={SMPC_COORDINATOR_QUEUE_PORT} "
        "-e REDIS_PSWD=agora "
        f"-e DB_URL={ip}:{SMPC_COORDINATOR_DB_PORT} "
        "-e DB_UNAME=sysadmin "
        "-e DB_PSWD=123qwe "
    )
    cmd = f"""docker run -d -p {container_ports} {env_variables} --name {SMPC_COORDINATOR_NAME} {image} {container_cmd}"""
    run(c, cmd)


@task
def start_smpc_coordinator(
    c, ip=None, smpc_image=None, smpc_db_image=None, smpc_queue_image=None
):
    """
    (Re)Creates all needed SMPC coordinator containers. If the containers exist, it will remove them and create them again.

    :param ip: The ip to use for container communication. If not set, it will read it from the `DEPLOYMENT_CONFIG_FILE`.
    :param smpc_image: The coordinator image to deploy. If not set, it will read it from the `DEPLOYMENT_CONFIG_FILE`.
    :param smpc_db_image: The db image to deploy. If not set, it will read it from the `DEPLOYMENT_CONFIG_FILE`.
    :param smpc_queue_image: The queue image to deploy. If not set, it will read it from the `DEPLOYMENT_CONFIG_FILE`.
    """

    if not ip:
        ip = get_deployment_config("ip")
    if not smpc_image:
        smpc_image = get_deployment_config("smpc", subconfig="smpc_image")
    if not smpc_db_image:
        smpc_db_image = get_deployment_config("smpc", subconfig="db_image")
    if not smpc_queue_image:
        smpc_queue_image = get_deployment_config("smpc", subconfig="queue_image")

    get_docker_image(c, smpc_image)
    get_docker_image(c, smpc_db_image)
    get_docker_image(c, smpc_queue_image)

    rm_containers(c, container_name="smpc_coordinator")

    start_smpc_coordinator_db(c, smpc_db_image)
    start_smpc_coordinator_queue(c, smpc_queue_image)
    start_smpc_coordinator_container(c, ip, smpc_image)


def start_smpc_player(c, ip, id, image):
    name = f"{SMPC_PLAYER_BASE_NAME}_{id}"
    message(
        f"Starting container {name} ...",
        Level.HEADER,
    )
    container_cmd = f"python player.py {id}"  # SMPC player id cannot be alphanumeric
    env_variables = (
        f"-e PLAYER_REPO_0=http://{ip}:7000 "
        f"-e PLAYER_REPO_1=http://{ip}:7001 "
        f"-e PLAYER_REPO_2=http://{ip}:7002 "
        f"-e COORDINATOR_URL=http://{ip}:{SMPC_COORDINATOR_PORT} "
        f"-e DB_URL={ip}:{SMPC_COORDINATOR_DB_PORT} "
        "-e DB_UNAME=sysadmin "
        "-e DB_PSWD=123qwe "
        f"-e PORT={SMPC_PLAYER_BASE_PORT + id}"
    )
    container_ports = (
        f"-p {6000 + id}:{6000 + id} "
        f"-p {SMPC_PLAYER_BASE_PORT + id}:{7000 + id} "
        f"-p {14000 + id}:{14000 + id} "
    )  # SMPC player port is increasing using the player id
    cmd = f"""docker run -d {container_ports} {env_variables} --name {name} {image} {container_cmd}"""
    run(c, cmd)


@task
def start_smpc_players(c, ip=None, image=None):
    """
    (Re)Creates 3 SMPC player containers. If the containers exist, it will remove them and create them again.

    :param ip: The ip to use for container communication. If not set, it will read it from the `DEPLOYMENT_CONFIG_FILE`.
    :param image: The smpc player image to deploy. If not set, it will read it from the `DEPLOYMENT_CONFIG_FILE`.
    """

    if not ip:
        ip = get_deployment_config("ip")
    if not image:
        image = get_deployment_config("smpc", subconfig="smpc_image")

    get_docker_image(c, image)

    rm_containers(c, container_name="smpc_player")

    for i in range(3):
        start_smpc_player(c, ip, i, image)


def start_smpc_client(c, worker_id, ip, image):
    worker_config_file = WORKERS_CONFIG_DIR / f"{worker_id}.toml"
    with open(worker_config_file) as fp:
        worker_config = toml.load(fp)

    client_id = worker_config["smpc"]["client_id"]
    client_port = worker_config["smpc"]["client_address"].split(":")[
        2
    ]  # Get the port from the address e.g. 'http://172.17.0.1:9000'

    name = f"{SMPC_CLIENT_BASE_NAME}_{client_id}"
    message(
        f"Starting container {name} ...",
        Level.HEADER,
    )
    container_cmd = f"python client.py"
    env_variables = (
        f"-e PLAYER_REPO_0=http://{ip}:7000 "
        f"-e PLAYER_REPO_1=http://{ip}:7001 "
        f"-e PLAYER_REPO_2=http://{ip}:7002 "
        f"-e COORDINATOR_URL=http://{ip}:{SMPC_COORDINATOR_PORT} "
        f"-e ID={client_id} "
        f"-e PORT={client_port} "
    )
    container_ports = f"-p {client_port}:{client_port} "
    cmd = f"""docker run -d {container_ports} {env_variables} --name {name} {image} {container_cmd}"""
    run(c, cmd)


@task
def start_smpc_clients(c, ip=None, image=None):
    """
    (Re)Creates 3 SMPC player containers. If the containers exist, it will remove them and create them again.

    :param ip: The ip to use for container communication. If not set, it will read it from the `DEPLOYMENT_CONFIG_FILE`.
    :param image: The smpc player image to deploy. If not set, it will read it from the `DEPLOYMENT_CONFIG_FILE`.
    """

    if not ip:
        ip = get_deployment_config("ip")
    if not image:
        image = get_deployment_config("smpc", subconfig="smpc_image")

    get_docker_image(c, image)

    rm_containers(c, container_name="smpc_client")

    for worker_id in get_localworker_ids():
        start_smpc_client(c, worker_id, ip, image)


@task
def deploy_smpc(c, ip=None, smpc_image=None, smpc_db_image=None, smpc_queue_image=None):
    """
    (Re)Creates all needed SMPC containers. If the containers exist, it will remove them and create them again.

    :param ip: The ip to use for container communication. If not set, it will read it from the `DEPLOYMENT_CONFIG_FILE`.
    :param smpc_image: The coordinator image to deploy. If not set, it will read it from the `DEPLOYMENT_CONFIG_FILE`.
    :param smpc_db_image: The db image to deploy. If not set, it will read it from the `DEPLOYMENT_CONFIG_FILE`.
    :param smpc_queue_image: The queue image to deploy. If not set, it will read it from the `DEPLOYMENT_CONFIG_FILE`.
    """
    rm_containers(c, smpc=True)
    start_smpc_coordinator(c, ip, smpc_image, smpc_db_image, smpc_queue_image)
    start_smpc_players(c, ip, smpc_image)
    start_smpc_clients(c, ip, smpc_image)


@task(iterable=["db"])
def reload_udfio(c, db):
    """
    Used for reloading the udfio module inside the monetdb containers.

    :param db: The names of the monetdb containers.
    """
    dbs = db
    for db in dbs:
        sql_reload_query = """
CREATE OR REPLACE FUNCTION
reload_udfio()
RETURNS
INT
LANGUAGE PYTHON
{
    import udfio
    import importlib
    importlib.reload(udfio)
    return 0
};

SELECT reload_udfio();
        """
        command = f'docker exec -t {db} mclient db --statement "{sql_reload_query}"'
        run(c, command)


def run(
    c,
    cmd,
    attach_=False,
    wait=True,
    warn=False,
    raise_error=False,
    show_ok=True,
    env=None,
):
    if attach_:
        c.run(cmd, pty=True, env=env)
        return

    if not wait:
        # TODO disown=True will make c.run(..) return immediately
        c.run(cmd, disown=True, env=env)
        # TODO wait is False to get in here
        # nevertheless, it will wait (sleep) for 4 seconds here, why??
        spin_wheel(time=4)
        if show_ok:
            message("Ok", Level.SUCCESS)
        return

    # TODO this is supposed to run when wait=True, yet asynchronous=True
    promise = c.run(cmd, asynchronous=True, warn=warn, env=env)
    # TODO and then it blocks here, what is the point of asynchronous=True?
    spin_wheel(promise=promise)
    stderr = promise.runner.stderr
    if stderr and raise_error:
        raise UnexpectedExit(stderr)
    result = promise.join()
    if show_ok:
        message("Ok", Level.SUCCESS)
    return result


class Level(Enum):
    HEADER = ("cyan", 1)
    BODY = ("white", 2)
    SUCCESS = ("green", 1)
    ERROR = ("red", 1)
    WARNING = ("yellow", 1)


def message(msg, level=Level.BODY):
    if msg.endswith("..."):
        end = ""
    else:
        end = "\n"
    color, indent_level = level.value
    prfx = "  " * indent_level
    msg = colored(msg, color)
    print(indent(msg, prfx), end=end, flush=True)


wheel = cycle(r"-\|/")


def spin_wheel(promise=None, time=None):
    dt = 0.15
    if promise is not None:
        print("  ", end="")
        for frame in wheel:
            print(frame + "  ", sep="", end="", flush=True)
            sleep(dt)
            print("\b\b\b", sep="", end="", flush=True)
            if promise.runner.process_is_finished:
                print("\b\b", end="")
                break
    elif time:
        print("  ", end="")
        for frame in wheel:
            print(frame + "  ", sep="", end="", flush=True)
            sleep(dt)
            print("\b\b\b", sep="", end="", flush=True)
            time -= dt
            if time <= 0:
                print("\b\b", end="")
                break


def get_deployment_config(config, subconfig=None):
    if not Path(DEPLOYMENT_CONFIG_FILE).is_file():
        raise FileNotFoundError(
            f"Please provide a --{config} parameter or create a deployment config file '{DEPLOYMENT_CONFIG_FILE}'"
        )
    with open(DEPLOYMENT_CONFIG_FILE) as fp:
        if subconfig:
            return toml.load(fp)[config][subconfig]
        return toml.load(fp)[config]


def get_worker_ids(all_=False, worker=None):
    worker_ids = []
    if all_:
        for worker_config_file in listdir(WORKERS_CONFIG_DIR):
            filename = Path(worker_config_file).stem
            worker_ids.append(filename)
    elif worker:
        worker_config_file = WORKERS_CONFIG_DIR / f"{worker}.toml"
        if not Path(worker_config_file).is_file():
            message(
                f"The configuration file for worker '{worker}', does not exist in directory '{WORKERS_CONFIG_DIR}'",
                Level.ERROR,
            )
            sys.exit(1)
        filename = Path(worker_config_file).stem
        worker_ids.append(filename)
    else:
        message(
            "Please specify a worker using --worker <worker> or use --all",
            Level.WARNING,
        )
        sys.exit(1)

    return worker_ids


def get_localworker_ids():
    all_worker_ids = get_worker_ids(all_=True)
    local_worker_ids = []
    for worker_id in all_worker_ids:
        worker_config_file = WORKERS_CONFIG_DIR / f"{worker_id}.toml"
        with open(worker_config_file) as fp:
            worker_config = toml.load(fp)
        if worker_config["role"] == "LOCALWORKER":
            local_worker_ids.append(worker_id)
    return local_worker_ids


def get_docker_image(c, image, always_pull=False):
    """
    Fetches a docker image locally.

    :param image: The image to pull from dockerhub.
    :param always_pull: Will pull the image even if it exists locally.
    """

    cmd = f"docker images -q {image}"
    _, image_tag = image.split(":")
    result = run(c, cmd, show_ok=False)
    if result.stdout != "" and image_tag != "latest" and image_tag != "dev":
        return

    message(f"Pulling image {image} ...", Level.HEADER)
    cmd = f"docker pull {image}"
    run(c, cmd)

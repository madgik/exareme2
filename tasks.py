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
import concurrent.futures
import copy
import glob
import itertools
import json
import os
import pathlib
import shutil
import sys
import time
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
AGGREGATOR_CONFIG_DIR = PROJECT_ROOT / "configs" / "aggregator"
CONTROLLER_LOCALWORKERS_CONFIG_FILE = (
    PROJECT_ROOT / "configs" / "controller" / "localworkers_config.json"
)
CONTROLLER_CONFIG_TEMPLATE_FILE = (
    PROJECT_ROOT / "exareme2" / "controller" / "config.toml"
)
AGGREGATOR_CONFIG_TEMPLATE_FILE = (
    PROJECT_ROOT / "exareme2" / "aggregator" / "config.toml"
)
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
EXAREME2_AGGREGATOR_CONFIG_FILE = "EXAREME2_AGGREGATOR_CONFIG_FILE"
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

        worker_config["sqlite"]["db_name"] = worker["id"]
        worker_config["monetdb"]["ip"] = deployment_config["ip"]
        worker_config["monetdb"]["port"] = worker["monetdb_port"]
        worker_config["monetdb"]["local_username"] = worker["local_monetdb_username"]
        worker_config["monetdb"]["local_password"] = worker["local_monetdb_password"]
        worker_config["monetdb"]["public_username"] = worker["public_monetdb_username"]
        worker_config["monetdb"]["public_password"] = worker["public_monetdb_password"]
        worker_config["monetdb"]["public_password"] = worker["public_monetdb_password"]

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

    controller_config[
        "worker_landscape_aggregator_update_interval"
    ] = deployment_config["worker_landscape_aggregator_update_interval"]
    controller_config["flower_execution_timeout"] = deployment_config[
        "flower_execution_timeout"
    ]
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

    # Create the aggregator config file
    with open(AGGREGATOR_CONFIG_TEMPLATE_FILE) as fp:
        template_aggregator_config = toml.load(fp)
    aggregator_config = copy.deepcopy(template_aggregator_config)
    aggregator_config["host"] = deployment_config["ip"]
    aggregator_config["port"] = deployment_config["aggregator"]["port"]
    aggregator_config["max_workers"] = deployment_config["aggregator"]["max_workers"]
    aggregator_config["log_level"] = deployment_config["log_level"]

    AGGREGATOR_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    aggregator_config_file = AGGREGATOR_CONFIG_DIR / "aggregator.toml"
    with open(aggregator_config_file, "w+") as fp:
        toml.dump(aggregator_config, fp)


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

    :param worker: A list of workers for which it will create the MonetDB containers.
    :param image: The image to deploy. If not set, it will read it from the DEPLOYMENT_CONFIG_FILE.
    :param log_level: If not set, it will read it from the DEPLOYMENT_CONFIG_FILE.
    :param nclients: If not set, it will read it from the DEPLOYMENT_CONFIG_FILE.
    :param monetdb_memory_limit: Memory limit for MonetDB. If not set, it will read it from the DEPLOYMENT_CONFIG_FILE.

    If an image is not provided it will use the 'monetdb_image' field from the DEPLOYMENT_CONFIG_FILE,
    e.g., monetdb_image = "madgik/exareme2_db:dev1.2".

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

    # Mount the aggregator source directory so that the MonetDB container
    # has access to the aggregator modules needed by the Python UDF.
    aggregator_src = "/home/kfilippopolitis/Desktop/aggregator_for_monetdb/"
    volume_aggregator = f"-v {aggregator_src}:/home/udflib/aggregator_for_monetdb"
    # Set PYTHONPATH to include the aggregator folder
    py_path_env = "-e PYTHONPATH=/home/udflib/aggregator_for_monetdb"

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
        cmd = (
            f"docker run -d -P -p {container_ports} "
            f"-e SOFT_RESTART_MEMORY_LIMIT={monetdb_memory_limit * 0.7} "
            f"-e HARD_RESTART_MEMORY_LIMIT={monetdb_memory_limit * 0.85} "
            f"-e LOG_LEVEL={log_level} {monetdb_nclient_env_var} "
            f"-e MAX_MEMORY={monetdb_memory_limit * 1048576} {monetdb_nclient_env_var} "
            f"{py_path_env} "
            f"-v {udfio_full_path}:/home/udflib/udfio.py "
            f"-v {TEST_DATA_FOLDER}:{TEST_DATA_FOLDER} "
            f"{volume_aggregator} "
            f"--name {container_name} --memory={monetdb_memory_limit}m {image}"
        )
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
        cmd = f"""poetry run mipdb init {get_sqlite_path(worker)}"""
        run(c, cmd)


@task
def update_wla(c):
    url = "http://localhost:5000/wla"
    response = requests.post(url)
    if response.status_code != 200:
        raise Exception("Failed to update the wla")
    print("Successfully updated wla.")


@task(iterable=["port"])
def load_data(c, use_sockets=False, worker=None):
    """
    Load data into the specified DB from the 'TEST_DATA_FOLDER'.

    For each directory under TEST_DATA_FOLDER (which must contain a CDEsMetadata.json),
    we build dictionaries mapping workers to a list of CSV files (and associated metadata)
    that they should load. Regular CSVs are assigned to LOCALWORKERs while test CSVs
    are assigned to GLOBALWORKERs.

    All load tasks are then executed concurrently.

    :param use_sockets: Flag determining if the data will be loaded via sockets.
    :param worker: If provided, only this worker's config is used.
    """

    def get_worker_configs():
        config_files = [WORKERS_CONFIG_DIR / f for f in listdir(WORKERS_CONFIG_DIR)]
        if not config_files:
            # No message printed
            sys.exit(1)
        configs = []
        for file in config_files:
            with open(file) as fp:
                configs.append(toml.load(fp))
        return configs

    def filter_worker_configs(worker_configs, worker, node_type):
        return [
            (conf["identifier"], conf["monetdb"]["port"])
            for conf in worker_configs
            if (not worker or conf["identifier"] == worker)
            and conf["role"] == node_type
        ]

    def load_data_model_metadata(c, cdes_file, worker_id_and_ports):
        """
        Run the add-data-model command for each worker in worker_id_and_ports.
        Return the (code, version) tuple.
        """
        with open(cdes_file) as fp:
            metadata = json.load(fp)
        data_model_code = metadata["code"]
        data_model_version = metadata["version"]

        def run_with_retries(c, cmd, retries=5, wait_seconds=1):
            attempt = 0
            while attempt < retries:
                try:
                    run(c, cmd, show_ok=False)
                    return
                except Exception as e:
                    attempt += 1
                    if attempt < retries:
                        time.sleep(wait_seconds)
                    else:
                        raise e

        for worker_id, port in worker_id_and_ports:
            cmd = (
                f"poetry run mipdb add-data-model {cdes_file} "
                f"{get_monetdb_configs_in_mipdb_format(port)} {get_sqlite_path(worker_id)}"
            )
            run_with_retries(c, cmd)
        return data_model_code, data_model_version

    def submit_load_task(
        executor,
        c,
        csv,
        data_model_code,
        data_model_version,
        worker_id,
        port,
        use_sockets,
    ):
        cmd = (
            f"poetry run mipdb add-dataset {csv} -d {data_model_code} -v {data_model_version} "
            f"--copy_from_file {not use_sockets} {get_monetdb_configs_in_mipdb_format(port)} {get_sqlite_path(worker_id)}"
        )
        return executor.submit(run, c, cmd, show_ok=False)

    # --- Build the assignment dictionaries ---
    # These dictionaries map (worker_id, port) to a list of tuples: (csv, data_model_code, data_model_version)
    local_tasks = {}  # for regular CSVs
    global_tasks = {}  # for test CSVs

    worker_configs = get_worker_configs()
    local_workers = filter_worker_configs(worker_configs, worker, "LOCALWORKER")
    if not local_workers:
        raise Exception("Local worker config files cannot be loaded.")
    global_workers = filter_worker_configs(worker_configs, worker, "GLOBALWORKER")
    if not global_workers:
        raise Exception("Global worker config files cannot be loaded.")

    # Iterate over directories under TEST_DATA_FOLDER
    for dirpath, _, filenames in os.walk(TEST_DATA_FOLDER):
        if "CDEsMetadata.json" not in filenames:
            continue
        cdes_file = os.path.join(dirpath, "CDEsMetadata.json")
        # Load metadata and run add-data-model for local workers
        data_model_code, data_model_version = load_data_model_metadata(
            c, cdes_file, local_workers
        )

        # Collect regular CSV files (exclude those ending with "test.csv")
        regular_csvs = [
            os.path.join(dirpath, f)
            for f in filenames
            if f.endswith(".csv") and not f.endswith("test.csv")
        ]
        # Collect test CSV files
        test_csvs = [
            os.path.join(dirpath, f) for f in filenames if f.endswith("test.csv")
        ]

        # For local workers: if only one, assign all regular CSVs to it;
        # if multiple, assign first set (files ending with "0.csv" but not "10.csv") to the first worker,
        # and distribute the rest round-robin among the others.
        if len(local_workers) == 1:
            worker_key = local_workers[0]
            local_tasks.setdefault(worker_key, [])
            for csv in regular_csvs:
                local_tasks[worker_key].append(
                    (csv, data_model_code, data_model_version)
                )
        else:
            first_worker = local_workers[0]
            local_tasks.setdefault(first_worker, [])
            first_csvs = sorted(
                [
                    csv
                    for csv in regular_csvs
                    if csv.endswith("0.csv") and not csv.endswith("10.csv")
                ]
            )
            local_tasks[first_worker].extend(
                [(csv, data_model_code, data_model_version) for csv in first_csvs]
            )
            remaining_csvs = sorted(
                [csv for csv in regular_csvs if csv not in first_csvs]
            )
            worker_cycle = itertools.cycle(local_workers[1:])
            for csv in remaining_csvs:
                worker_key = next(worker_cycle)
                local_tasks.setdefault(worker_key, []).append(
                    (csv, data_model_code, data_model_version)
                )

        # For test CSVs, assign them to the first global worker.
        global_worker = global_workers[0]
        global_tasks.setdefault(global_worker, [])
        for csv in test_csvs:
            global_tasks[global_worker].append(
                (csv, data_model_code, data_model_version)
            )

    # --- Schedule all tasks concurrently ---
    all_tasks = []
    max_parallel = 10  # adjust as needed
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
        # Schedule local tasks
        for worker_key, tasks in local_tasks.items():
            worker_id, port = worker_key
            for csv, data_model_code, data_model_version in tasks:
                all_tasks.append(
                    submit_load_task(
                        executor,
                        c,
                        csv,
                        data_model_code,
                        data_model_version,
                        worker_id,
                        port,
                        use_sockets,
                    )
                )
        # Schedule global (test) tasks
        for worker_key, tasks in global_tasks.items():
            worker_id, port = worker_key
            for csv, data_model_code, data_model_version in tasks:
                cmd = (
                    f"poetry run mipdb add-dataset {csv} -d {data_model_code} -v {data_model_version} "
                    f"--copy_from_file {not use_sockets} {get_monetdb_configs_in_mipdb_format(port)} {get_sqlite_path(worker_id)}"
                )
                all_tasks.append(executor.submit(run, c, cmd, show_ok=False))
        concurrent.futures.wait(all_tasks)
        # Final message is now disabled or can be removed if desired.
        message("All data loading tasks completed.", Level.SUCCESS)


def get_sqlite_path(worker_id):
    return f"--sqlite_db_path {TEST_DATA_FOLDER}/{worker_id}.db"


def get_monetdb_configs_in_mipdb_format(port):
    return (
        f"--ip 127.0.0.1 "
        f"--port {port} "
        f"--username admin "
        f"--password executor "
        f"--db_name db"
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
def kill_aggregator(c):
    """
    Kill the aggregator service by finding and terminating its process.

    This task looks for any process whose command line contains 'grpc_agg_server.py'
    and terminates it.
    """
    res = run(c, "ps aux | grep '[g]rpc_agg_server.py'", warn=True, show_ok=False)
    if res.ok and res.stdout.strip():
        message("Killing aggregator process...", Level.HEADER)
        cmd = "ps aux | grep '[g]rpc_agg_server.py' | awk '{print $2}' | xargs kill -9"
        run(c, cmd)
        message("Aggregator process killed.", Level.SUCCESS)
    else:
        message("No aggregator process found.", Level.HEADER)


@task
def start_aggregator(c, detached=False):
    """
    Starts the Aggregation gRPC server.

    The aggregator now uses the configuration provided in the config file
    (via exareme2.aggregator.config) so no command-line parameters are needed.
    If detached is True, the server will run in the background.
    """

    kill_aggregator(c)
    message("Starting Aggregator...", Level.HEADER)
    aggregator_config_file = AGGREGATOR_CONFIG_DIR / "aggregator.toml"

    server_script = PROJECT_ROOT / "exareme2" / "aggregator" / "grpc_agg_server.py"

    if not server_script.exists():
        message(f"Aggregator server script not found at {server_script}.", Level.ERROR)
        return

    with c.prefix(f"export {EXAREME2_AGGREGATOR_CONFIG_FILE}={aggregator_config_file}"):
        # Command simply runs the aggregator server script.
        cmd = f"PYTHONPATH={PROJECT_ROOT} python {server_script}"

        if detached:
            log_file = OUTDIR / "aggregator_server.out"
            c.run(f"{cmd} >> {log_file} 2>&1 &", disown=True)
            message("Ok", Level.SUCCESS)
        else:
            c.run(cmd, pty=True)


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
    start_all=True,
    start_controller_=False,
    start_aggregator_=False,
    start_workers=False,
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
    :param start_all: Start all worker/controller api flag.
    :param start_controller_: Start controller api flag.
    :param start_workers: Start all workers flag.
    :param log_level: Used for the dev logs. If not provided, it looks in the `DEPLOYMENT_CONFIG_FILE`.
    :param framework_log_level: Used for the engine api. If not provided, it looks in the `DEPLOYMENT_CONFIG_FILE`.
    :param monetdb_image: Used for the db containers. If not provided, it looks in the `DEPLOYMENT_CONFIG_FILE`.
    :param monetdb_nclients: Used for the db containers. If not provided, it looks in the `DEPLOYMENT_CONFIG_FILE`.
    :param exareme2_algorithm_folders: Used from the api. If not provided, it looks in the `DEPLOYMENT_CONFIG_FILE`.
    :param flower_algorithm_folders: Used from the api. If not provided, it looks in the `DEPLOYMENT_CONFIG_FILE`.
    :param smpc: Deploy the SMPC cluster as well. If not provided, it looks in the `DEPLOYMENT_CONFIG_FILE`.
    """

    if not log_level:
        log_level = get_deployment_config("log_level")

    if not framework_log_level:
        framework_log_level = get_deployment_config("framework_log_level")

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

    worker_ids.sort()  # Sorting the ids protects removing a similarly named id, localworker1 would remove localworker10.

    create_monetdb(
        c,
        worker=worker_ids,
        image=monetdb_image,
        log_level=log_level,
        nclients=monetdb_nclients,
    )
    init_system_tables(c, worker=worker_ids)
    create_rabbitmq(c, worker=worker_ids)

    if start_workers or start_all:
        start_worker(
            c,
            all_=True,
            framework_log_level=framework_log_level,
            detached=True,
            exareme2_algorithm_folders=exareme2_algorithm_folders,
            flower_algorithm_folders=flower_algorithm_folders,
            exaflow_algorithm_folders=exaflow_algorithm_folders,
        )

    if start_aggregator_ or start_all:
        start_aggregator(c, detached=True)

    # Start CONTROLLER service
    if start_controller_ or start_all:
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
    kill_aggregator(c)
    kill_worker(c, all_=True)
    rm_containers(c, monetdb=True, rabbitmq=True, smpc=True)

    # Create a pattern for .db files
    pattern = os.path.join(TEST_DATA_FOLDER, "*.db")

    # Delete each .db file
    for sqlite_path in glob.glob(pattern):
        clean_sqlite(sqlite_path)
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

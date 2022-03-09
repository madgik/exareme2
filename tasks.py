"""
Deployment script used for the development of the MIP-Engine.

In order to understand this script a basic knowledge of the system is required, this script
does not contain the documentation of the engine. The documentation of the tasks,
in this script, is targeted to the specifics of the development deployment process.

This script deploys all the containers and services natively on your machine.
It deploys the containers on different ports and then configures the services to use the appropriate ports.

A node service uses a configuration file either on the default location './mipengine/node/config.toml'
or in the location of the env variable 'MIPENGINE_NODE_CONFIG_FILE', if the env variable is set.
This deployment script used for development, uses the env variable logic, therefore before deploying each
node service the env variable is changed to the location of the node services' config file.

In order for this script's tasks to work the './configs/nodes' folder should contain all the node's config files
following the './mipengine/node/config.toml' as template.
You can either create the files manually or using a '.deployment.toml' file with the following template
```
ip = "172.17.0.1"
log_level = "INFO"
framework_log_level ="INFO"
monetdb_image = "madgik/mipenginedb:dev1.3"

[[nodes]]
id = "globalnode"
monetdb_port=50000
rabbitmq_port=5670

[[nodes]]
id = "localnode1"
monetdb_port=50001
rabbitmq_port=5671

[[nodes]]
id = "localnode2"
monetdb_port=50002
rabbitmq_port=5672
```

and by running the command 'inv create-configs'.

The node services are named after their config file. If a config file is named './configs/nodes/localnode1.toml'
the node service will be called 'localnode1' and should be referenced using that in the following tasks.

Paths are subject to change so in the following documentation the global variables will be used.

"""
import itertools
import json
import pathlib
import shutil
import sys
from enum import Enum
from itertools import cycle
from os import listdir
from os import path
from pathlib import Path
from textwrap import indent
from time import sleep

import toml
from invoke import UnexpectedExit
from invoke import task
from termcolor import colored

import tests
from mipengine.udfgen import udfio

PROJECT_ROOT = Path(__file__).parent
DEPLOYMENT_CONFIG_FILE = PROJECT_ROOT / ".deployment.toml"
NODES_CONFIG_DIR = PROJECT_ROOT / "configs" / "nodes"
NODE_CONFIG_TEMPLATE_FILE = PROJECT_ROOT / "mipengine" / "node" / "config.toml"
CONTROLLER_CONFIG_DIR = PROJECT_ROOT / "configs" / "controller"
CONTROLLER_LOCALNODES_CONFIG_FILE = (
    PROJECT_ROOT / "configs" / "controller" / "localnodes_config.json"
)
CONTROLLER_CONFIG_TEMPLATE_FILE = (
    PROJECT_ROOT / "mipengine" / "controller" / "config.toml"
)
OUTDIR = Path("/tmp/mipengine/")
if not OUTDIR.exists():
    OUTDIR.mkdir()

TEST_DATA_FOLDER = Path(tests.__file__).parent / "test_data"

ALGORITHM_FOLDERS_ENV_VARIABLE = "ALGORITHM_FOLDERS"
MIPENGINE_NODE_CONFIG_FILE = "MIPENGINE_NODE_CONFIG_FILE"

# TODO Add pre-tasks when this is implemented https://github.com/pyinvoke/invoke/issues/170
# Right now if we call a task from another task, the "pre"-task is not executed


@task
def create_configs(c):
    """
    Create the node and controller services config files, using 'DEPLOYMENT_CONFIG_FILE'.
    """
    if path.exists(NODES_CONFIG_DIR) and path.isdir(NODES_CONFIG_DIR):
        shutil.rmtree(NODES_CONFIG_DIR)
    NODES_CONFIG_DIR.mkdir(parents=True)

    if not Path(DEPLOYMENT_CONFIG_FILE).is_file():
        raise FileNotFoundError(
            f"Deployment config file '{DEPLOYMENT_CONFIG_FILE}' not found."
        )

    with open(DEPLOYMENT_CONFIG_FILE) as fp:
        deployment_config = toml.load(fp)

    with open(NODE_CONFIG_TEMPLATE_FILE) as fp:
        template_node_config = toml.load(fp)

    for node in deployment_config["nodes"]:
        node_config = template_node_config.copy()

        node_config["cdes_metadata_path"] = deployment_config["cdes_metadata_path"]

        node_config["identifier"] = node["id"]
        node_config["role"] = node["role"]
        node_config["log_level"] = deployment_config["log_level"]
        node_config["framework_log_level"] = deployment_config["framework_log_level"]

        node_config["monetdb"]["ip"] = deployment_config["ip"]
        node_config["monetdb"]["port"] = node["monetdb_port"]

        node_config["rabbitmq"]["ip"] = deployment_config["ip"]
        node_config["rabbitmq"]["port"] = node["rabbitmq_port"]

        node_config["smpc"]["enabled"] = deployment_config["smpc"]["enabled"]
        node_config["smpc"]["optional"] = deployment_config["smpc"]["optional"]

        node_config_file = NODES_CONFIG_DIR / f"{node['id']}.toml"
        with open(node_config_file, "w+") as fp:
            toml.dump(node_config, fp)

    # Create the controller config file
    with open(CONTROLLER_CONFIG_TEMPLATE_FILE) as fp:
        template_controller_config = toml.load(fp)
    controller_config = template_controller_config.copy()
    controller_config["log_level"] = deployment_config["log_level"]
    controller_config["framework_log_level"] = deployment_config["framework_log_level"]

    controller_config["cdes_metadata_path"] = deployment_config["cdes_metadata_path"]
    controller_config["node_registry_update_interval"] = deployment_config[
        "node_registry_update_interval"
    ]
    controller_config["rabbitmq"]["celery_tasks_timeout"] = deployment_config[
        "celery_tasks_timeout"
    ]
    controller_config["deployment_type"] = "LOCAL"

    controller_config["localnodes"]["config_file"] = str(
        CONTROLLER_LOCALNODES_CONFIG_FILE
    )
    controller_config["localnodes"]["dns"] = ""
    controller_config["localnodes"]["port"] = ""

    controller_config["smpc"]["enabled"] = deployment_config["smpc"]["enabled"]
    controller_config["smpc"]["optional"] = deployment_config["smpc"]["optional"]

    CONTROLLER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    controller_config_file = CONTROLLER_CONFIG_DIR / "controller.toml"
    with open(controller_config_file, "w+") as fp:
        toml.dump(controller_config, fp)

    # Create the controller localnodes config file
    localnodes_addresses = [
        f"{deployment_config['ip']}:{node['rabbitmq_port']}"
        for node in deployment_config["nodes"]
    ]
    with open(CONTROLLER_LOCALNODES_CONFIG_FILE, "w+") as fp:
        json.dump(localnodes_addresses, fp)


@task
def install_dependencies(c):
    """Install project dependencies using poetry."""
    message("Installing dependencies...", Level.HEADER)
    cmd = "poetry install"
    run(c, cmd)


@task
def rm_containers(c, container_name=None, monetdb=False, rabbitmq=False):
    """
    Remove the specified docker containers, either by container or relative name.

    :param container_name: If set, removes the container with the specified name.
    :param monetdb: If True, it will remove all monetdb containers.
    :param rabbitmq: If True, it will remove all rabbitmq containers.

    If nothing is set, nothing is removed.
    """
    names = []
    if monetdb:
        names.append("monetdb")
    if rabbitmq:
        names.append("rabbitmq")
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
            message(f"Removing {name} container...", Level.HEADER)
            cmd = f"docker rm -vf $(docker ps -qa --filter name={name})"
            run(c, cmd)
        else:
            message(f"No {name} container to remove", level=Level.HEADER)


@task(iterable=["node"])
def create_monetdb(c, node, image=None, log_level=None):
    """
    (Re)Create MonetDB container(s) for given node(s). If the container exists, it will remove it and create it again.

    :param node: A list of nodes for which it will create the monetdb containers.
    :param image: The image to deploy. If not set, it will read it from the `DEPLOYMENT_CONFIG_FILE`.
    :param log_level: If not set, it will read it from the `DEPLOYMENT_CONFIG_FILE`.

    If an image is not provided it will use the 'monetdb_image' field from
    the 'DEPLOYMENT_CONFIG_FILE' ex. monetdb_image = "madgik/mipenginedb:dev1.2"

    The data of the monetdb container are not persisted. If the container is recreated, all data will be lost.
    """
    if not node:
        message("Please specify a node using --node <node>", Level.WARNING)
        sys.exit(1)

    if not image:
        image = get_deployment_config("monetdb_image")

    if not log_level:
        log_level = get_deployment_config("log_level")

    get_docker_image(c, image)

    udfio_full_path = path.abspath(udfio.__file__)

    node_ids = node
    for node_id in node_ids:
        container_name = f"monetdb-{node_id}"
        rm_containers(c, container_name=container_name)

        node_config_file = NODES_CONFIG_DIR / f"{node_id}.toml"
        with open(node_config_file) as fp:
            node_config = toml.load(fp)
        container_ports = f"{node_config['monetdb']['port']}:50000"
        message(
            f"Starting container {container_name} on ports {container_ports}...",
            Level.HEADER,
        )
        # A volume is used to pass the udfio inside the monetdb container.
        # This is done so that we don't need to rebuild every time the udfio.py file is changed.
        cmd = f"""docker run -d -P -p {container_ports} -e LOG_LEVEL={log_level} -v {udfio_full_path}:/home/udflib/udfio.py --name {container_name} {image}"""
        run(c, cmd)


@task(iterable=["port"])
def init_monetdb(c, port):
    """
    Initialize MonetDB container(s) with mipdb.

    :param port: A list of container ports that will be initialized.
    """
    ports = port
    for port in ports:
        message(
            f"Initializing MonetDB with mipdb in port: {port}...",
            Level.HEADER,
        )
        cmd = f"""poetry run mipdb init --ip 172.17.0.1 --port {port}"""
        run(c, cmd)


@task(iterable=["port"])
def load_data(c, port=None):
    """
    Load data into the specified DB from the 'TEST_DATA_FOLDER'.

    :param port: A list of ports, in which it will load the data. If not set, it will use the `NODES_CONFIG_DIR` files.
    """

    local_node_ports = port
    if not local_node_ports:
        config_files = [NODES_CONFIG_DIR / file for file in listdir(NODES_CONFIG_DIR)]
        if not config_files:
            message(
                f"There are no node config files to be used for data import. Folder: {NODES_CONFIG_DIR}",
                Level.WARNING,
            )
            sys.exit(1)

        local_node_ports = []
        for node_config_file in config_files:
            with open(node_config_file) as fp:
                node_config = toml.load(fp)
            if node_config["role"] == "LOCALNODE":
                local_node_ports.append(node_config["monetdb"]["port"])

    # Load the test data folder into the dbs
    data_model_folders = [
        TEST_DATA_FOLDER / folder for folder in listdir(TEST_DATA_FOLDER)
    ]
    local_node_ports_cycle = itertools.cycle(local_node_ports)
    for data_model_folder in data_model_folders:

        # Load all data models in each db
        with open(data_model_folder / "CDEsMetadata.json") as data_model_metadata_file:
            data_model_metadata = json.load(data_model_metadata_file)
            data_model_code = data_model_metadata["code"]
            data_model_version = data_model_metadata["version"]
        cdes_file = data_model_folder / "CDEsMetadata.json"
        for port in local_node_ports:
            message(
                f"Loading data model '{data_model_code}:{data_model_version}' metadata in MonetDB at port {port}...",
                Level.HEADER,
            )
            cmd = f"poetry run mipdb add-data-model {cdes_file} --port {port} "
            run(c, cmd)

        # Load the data model's csvs in the nodes with round-robin fashion
        csvs = sorted(
            [
                data_model_folder / file
                for file in listdir(data_model_folder)
                if file.endswith(".csv")
            ]
        )
        for csv in csvs:
            port = next(local_node_ports_cycle)
            message(
                f"Loading dataset {pathlib.PurePath(csv).name} in MonetDB at port {port}...",
                Level.HEADER,
            )
            cmd = f"poetry run mipdb add-dataset {csv} -d {data_model_code} -v {data_model_version} --port {port} "
            run(c, cmd)


@task(iterable=["node"])
def create_rabbitmq(c, node, rabbitmq_image=None):
    """
    (Re)Create RabbitMQ container(s) of given node(s). If the container exists, remove it and create it again.

    :param node: A list of nodes for which to (re)create the rabbitmq containers.
    :param rabbitmq_image: The image to deploy. If not set, it will read it from the `DEPLOYMENT_CONFIG_FILE`.

    """
    if not node:
        message("Please specify a node using --node <node>", Level.WARNING)
        sys.exit(1)

    if not rabbitmq_image:
        rabbitmq_image = get_deployment_config("rabbitmq_image")

    get_docker_image(c, rabbitmq_image)

    node_ids = node
    for node_id in node_ids:
        container_name = f"rabbitmq-{node_id}"
        rm_containers(c, container_name=container_name)

        node_config_file = NODES_CONFIG_DIR / f"{node_id}.toml"
        with open(node_config_file) as fp:
            node_config = toml.load(fp)
        container_ports = f"{node_config['rabbitmq']['port']}:5672"
        message(
            f"Starting container {container_name} on ports {container_ports}...",
            Level.HEADER,
        )
        cmd = f"docker run -d -p {container_ports} --name {container_name} {rabbitmq_image}"
        run(c, cmd)

    for node_id in node_ids:
        container_name = f"rabbitmq-{node_id}"

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
def kill_node(c, node=None, all_=False):
    """
    Kill the node(s) service(s).

    :param node: The node service to kill.
    :param all_: If set, all node services will be killed.
    """

    if all_:
        node_pattern = ""
    elif node:
        node_pattern = node
    else:
        message("Please specify a node using --node <node> or use --all", Level.WARNING)
        sys.exit(1)

    res_bin = run(
        c,
        f"ps aux | grep '[c]elery' | grep 'worker' | grep '{node_pattern}' ",
        warn=True,
        show_ok=False,
    )

    if res_bin.ok:
        message(
            f"Killing previous celery instance(s) with pattern '{node_pattern}' ...",
            Level.HEADER,
        )
        # In order for the node service to be killed, we need to kill the celery worker process with the "node_pattern"
        # in it's name and it's parent process, the celery main process.
        cmd = (
            f"pid=$(ps aux | grep '[c]elery' | grep 'worker' | grep '{node_pattern}' | awk '{{print $2}}') "
            f"&& pgrep -P $pid | xargs kill -9 "
            f"&& kill -9 $pid "
        )
        run(c, cmd, warn=True)
    else:
        message("No celery instances found", Level.HEADER)


@task
def start_node(
    c,
    node=None,
    all_=False,
    framework_log_level=None,
    detached=False,
    algorithm_folders=None,
):
    """
    (Re)Start the node(s) service(s). If a node service is running, stop and start it again.

    :param node: The node to start, using the proper file in the `NODES_CONFIG_DIR`.
    :param all_: If set, the nodes of which the configuration file exists, will be started.
    :param framework_log_level: If not provided, it will look into the `DEPLOYMENT_CONFIG_FILE`.
    :param detached: If set to True, it will start the service in the background.
    :param algorithm_folders: Used from the services. If not provided, it looks in the `DEPLOYMENT_CONFIG_FILE`.

    The containers related to the services remain unchanged.
    """

    if not framework_log_level:
        framework_log_level = get_deployment_config("framework_log_level")

    if not algorithm_folders:
        algorithm_folders = get_deployment_config("algorithm_folders")
    if not isinstance(algorithm_folders, str):
        raise ValueError(
            "The algorithm_folders configuration must be a comma separated string."
        )

    node_ids = get_node_ids(all_, node)

    for node_id in node_ids:
        kill_node(c, node_id)

        message(f"Starting Node {node_id}...", Level.HEADER)
        node_config_file = NODES_CONFIG_DIR / f"{node_id}.toml"
        with c.prefix(f"export {ALGORITHM_FOLDERS_ENV_VARIABLE}={algorithm_folders}"):
            with c.prefix(f"export {MIPENGINE_NODE_CONFIG_FILE}={node_config_file}"):
                outpath = OUTDIR / (node_id + ".out")
                if detached or all_:
                    cmd = (
                        f"PYTHONPATH={PROJECT_ROOT} poetry run celery "
                        f"-A mipengine.node.node worker -l {framework_log_level} >> {outpath} "
                        f"--purge 2>&1"
                    )
                    run(c, cmd, wait=False)
                else:
                    cmd = (
                        f"PYTHONPATH={PROJECT_ROOT} poetry run celery -A "
                        f"mipengine.node.node worker -l {framework_log_level} --purge"
                    )
                    run(c, cmd, attach_=True)


@task
def kill_controller(c):
    """Kill the controller service."""
    res = run(c, "ps aux | grep '[q]uart'", warn=True, show_ok=False)
    if res.ok:
        message("Killing previous Quart instances...", Level.HEADER)
        cmd = "ps aux | grep '[q]uart' | awk '{ print $2}' | xargs kill -9 && sleep 5"
        run(c, cmd)
    else:
        message("No quart instance found", Level.HEADER)


@task
def start_controller(c, detached=False, algorithm_folders=None):
    """
    (Re)Start the controller service. If the service is already running, stop and start it again.

    :param detached: If set to True, it will start the service in the background.
    :param algorithm_folders: Used from the services. If not provided, it looks in the `DEPLOYMENT_CONFIG_FILE`.
    """

    if not algorithm_folders:
        algorithm_folders = get_deployment_config("algorithm_folders")
    if not isinstance(algorithm_folders, str):
        raise ValueError(
            "The algorithm_folders configuration must be a comma separated string."
        )

    kill_controller(c)

    message("Starting Controller...", Level.HEADER)
    controller_config_file = CONTROLLER_CONFIG_DIR / "controller.toml"
    with c.prefix(f"export {ALGORITHM_FOLDERS_ENV_VARIABLE}={algorithm_folders}"):
        with c.prefix(
            f"export MIPENGINE_CONTROLLER_CONFIG_FILE={controller_config_file}"
        ):
            with c.prefix("export QUART_APP=mipengine/controller/api/app:app"):
                outpath = OUTDIR / "controller.out"
                if detached:
                    cmd = f"PYTHONPATH={PROJECT_ROOT} poetry run quart run --host=0.0.0.0>> {outpath} 2>&1"
                    run(c, cmd, wait=False)
                else:
                    cmd = (
                        f"PYTHONPATH={PROJECT_ROOT} poetry run quart run --host=0.0.0.0"
                    )
                    run(c, cmd, attach_=True)


@task
def deploy(
    c,
    install_dep=True,
    start_all=True,
    start_controller_=False,
    start_nodes=False,
    log_level=None,
    framework_log_level=None,
    monetdb_image=None,
    algorithm_folders=None,
):
    """
    Install dependencies, (re)create all the containers and (re)start all the services.

    :param install_dep: Install dependencies or not.
    :param start_all: Start all node/controller services flag.
    :param start_controller_: Start controller services flag.
    :param start_nodes: Start all nodes flag.
    :param log_level: Used for the dev logs. If not provided, it looks in the `DEPLOYMENT_CONFIG_FILE`.
    :param framework_log_level: Used for the engine services. If not provided, it looks in the `DEPLOYMENT_CONFIG_FILE`.
    :param monetdb_image: Used for the db containers. If not provided, it looks in the `DEPLOYMENT_CONFIG_FILE`.
    :param algorithm_folders: Used from the services.
    """

    if not log_level:
        log_level = get_deployment_config("log_level")

    if not framework_log_level:
        framework_log_level = get_deployment_config("framework_log_level")

    if not monetdb_image:
        monetdb_image = get_deployment_config("monetdb_image")

    if install_dep:
        install_dependencies(c)

    # start NODE services
    config_files = [NODES_CONFIG_DIR / file for file in listdir(NODES_CONFIG_DIR)]
    if not config_files:
        message(
            f"There are no node config files to be used for deployment. Folder: {NODES_CONFIG_DIR}",
            Level.WARNING,
        )
        sys.exit(1)

    node_ids = []
    monetdb_ports = []
    for node_config_file in config_files:
        with open(node_config_file) as fp:
            node_config = toml.load(fp)
        node_ids.append(node_config["identifier"])
        monetdb_ports.append(node_config["monetdb"]["port"])

    create_monetdb(c, node=node_ids, image=monetdb_image, log_level=log_level)
    create_rabbitmq(c, node=node_ids)
    init_monetdb(c, port=monetdb_ports)

    if start_nodes or start_all:
        start_node(
            c,
            all_=True,
            framework_log_level=framework_log_level,
            detached=True,
            algorithm_folders=algorithm_folders,
        )

    # start CONTROLLER service
    if start_controller_ or start_all:
        start_controller(c, detached=True, algorithm_folders=algorithm_folders)


@task
def attach(c, node=None, controller=False, db=None):
    """
    Attach to a node/controller service or a db container.

    :param node: The node service name to which to attach.
    :param controller: Attach to controller flag.
    :param db: The db container name to which to attach.
    """
    if (node or controller) and not (node and controller):
        fname = node or "controller"
        outpath = OUTDIR / (fname + ".out")
        cmd = f"tail -f {outpath}"
        run(c, cmd, attach_=True)
    elif db:
        run(c, f"docker exec -it {db} mclient db", attach_=True)
    else:
        message("You must attach to Node, Controller or DB", Level.WARNING)
        sys.exit(1)


@task
def cleanup(c):
    """Kill all node/controller services and remove all monetdb/rabbitmq containers."""
    kill_controller(c)
    kill_node(c, all_=True)
    rm_containers(c, monetdb=True, rabbitmq=True)
    if OUTDIR.exists():
        message(f"Removing {OUTDIR}...", level=Level.HEADER)
        for outpath in OUTDIR.glob("*.out"):
            outpath.unlink()
        OUTDIR.rmdir()
        message("Ok", level=Level.SUCCESS)


@task
def start_flower(c, node=None, all_=False):
    """
    (Re)Start flower monitoring tool. If flower is already running, stop ir and start it again.

    :param node: The node service, for which to create the flower monitoring.
    :param all_: If set, it will create monitoring for all node services in the `NODES_CONFIG_DIR`.
    """

    kill_all_flowers(c)

    FLOWER_PORT = 5550

    node_ids = get_node_ids(all_, node)

    for node_id in node_ids:
        node_config_file = NODES_CONFIG_DIR / f"{node_id}.toml"
        with open(node_config_file) as fp:
            node_config = toml.load(fp)

        ip = node_config["rabbitmq"]["ip"]
        port = node_config["rabbitmq"]["port"]
        user_and_password = (
            node_config["rabbitmq"]["user"] + ":" + node_config["rabbitmq"]["password"]
        )
        vhost = node_config["rabbitmq"]["vhost"]
        flower_url = ip + ":" + str(port)
        broker_api = f"amqp://{user_and_password}@{flower_url}/{vhost}"

        flower_index = node_ids.index(node_id)
        flower_port = FLOWER_PORT + flower_index

        message(f"Starting flower container for node {node_id}...", Level.HEADER)
        command = f"docker run --name flower-{node_id} -d -p {flower_port}:5555 mher/flower:0.9.5 flower --broker={broker_api} "
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


def run(c, cmd, attach_=False, wait=True, warn=False, raise_error=False, show_ok=True):
    if attach_:
        c.run(cmd, pty=True)
        return

    if not wait:
        # TODO disown=True will make c.run(..) return immediately
        c.run(cmd, disown=True)
        # TODO wait is False to get in here
        # nevertheless, it will wait (sleep) for 4 seconds here, why??
        spin_wheel(time=4)
        if show_ok:
            message("Ok", Level.SUCCESS)
        return

    # TODO this is supposed to run when wait=True, yet asynchronous=True
    promise = c.run(cmd, asynchronous=True, warn=warn)
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


def get_deployment_config(config):
    if not Path(DEPLOYMENT_CONFIG_FILE).is_file():
        raise FileNotFoundError(
            f"Please provide a --{config} parameter or create a deployment config file '{DEPLOYMENT_CONFIG_FILE}'"
        )

    with open(DEPLOYMENT_CONFIG_FILE) as fp:
        return toml.load(fp)[config]


def get_node_ids(all_=False, node=None):
    node_ids = []
    if all_:
        for node_config_file in listdir(NODES_CONFIG_DIR):
            filename = Path(node_config_file).stem
            node_ids.append(filename)
    elif node:
        node_config_file = NODES_CONFIG_DIR / f"{node}.toml"
        if not Path(node_config_file).is_file():
            message(
                f"The configuration file for node '{node}', does not exist in directory '{NODES_CONFIG_DIR}'",
                Level.ERROR,
            )
            sys.exit(1)
        filename = Path(node_config_file).stem
        node_ids.append(filename)
    else:
        message("Please specify a node using --node <node> or use --all", Level.WARNING)
        sys.exit(1)

    return node_ids


def get_docker_image(c, image, always_pull=False):
    """
    Fetches a docker image locally.

    :param image: The image to pull from dockerhub.
    :param always_pull: Will pull the image even if it exists locally.
    """

    cmd = f"docker images -q {image}"
    _, image_tag = image.split(":")
    result = run(c, cmd, show_ok=False)
    if result.stdout != "" and image_tag != "latest":
        return

    message(f"Pulling image {image} ...", Level.HEADER)
    cmd = f"docker pull {image}"
    run(c, cmd)

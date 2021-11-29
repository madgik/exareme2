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
celery_log_level ="INFO"
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
import json
import sys
from enum import Enum
from itertools import cycle
from os import listdir
from pathlib import Path
from textwrap import indent
from time import sleep

import toml
from invoke import UnexpectedExit
from invoke import task
from termcolor import colored

import tests

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

DEMO_DATA_FOLDER = Path(tests.__file__).parent / "demo_data"

ALGORITHM_FOLDERS_ENV_VARIABLE = "ALGORITHM_FOLDERS"
MIPENGINE_NODE_CONFIG_FILE = "MIPENGINE_NODE_CONFIG_FILE"

# TODO Add pre-tasks when this is implemented https://github.com/pyinvoke/invoke/issues/170
# Right now if we call a task from another task, the "pre"-task is not executed


@task
def create_configs(c):
    """
    Create the node and controller services config files, using 'DEPLOYMENT_CONFIG_FILE'.
    """

    if not Path(DEPLOYMENT_CONFIG_FILE).is_file():
        raise FileNotFoundError(
            f"Deployment config file '{DEPLOYMENT_CONFIG_FILE}' not found."
        )

    with open(DEPLOYMENT_CONFIG_FILE) as fp:
        deployment_config = toml.load(fp)

    # Create the nodes config files
    with open(NODE_CONFIG_TEMPLATE_FILE) as fp:
        template_node_config = toml.load(fp)

    for node in deployment_config["nodes"]:
        node_config = template_node_config.copy()

        node_config["cdes_metadata_path"] = deployment_config["cdes_metadata_path"]

        node_config["identifier"] = node["id"]
        node_config["role"] = node["role"]
        node_config["log_level"] = deployment_config["log_level"]

        node_config["monetdb"]["ip"] = deployment_config["ip"]
        node_config["monetdb"]["port"] = node["monetdb_port"]

        node_config["rabbitmq"]["ip"] = deployment_config["ip"]
        node_config["rabbitmq"]["port"] = node["rabbitmq_port"]

        NODES_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        node_config_file = NODES_CONFIG_DIR / f"{node['id']}.toml"
        with open(node_config_file, "w+") as fp:
            toml.dump(node_config, fp)

    # Create the controller config file
    with open(CONTROLLER_CONFIG_TEMPLATE_FILE) as fp:
        template_controller_config = toml.load(fp)
    controller_config = template_controller_config.copy()
    controller_config["cdes_metadata_path"] = deployment_config["cdes_metadata_path"]
    controller_config["node_registry_update_interval"] = deployment_config[
        "node_registry_update_interval"
    ]

    controller_config["deployment_type"] = "LOCAL"

    controller_config["localnodes"]["config_file"] = str(
        CONTROLLER_LOCALNODES_CONFIG_FILE
    )
    controller_config["localnodes"]["dns"] = ""
    controller_config["localnodes"]["port"] = ""

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
def create_monetdb(c, node, monetdb_image=None):
    """
    (Re)Create MonetDB container(s) for given node(s). If the container exists, it will remove it and create it again.

    :param node: A list of nodes for which it will create the monetdb containers.
    :param monetdb_image: The image to deploy. If not set, it will read it from the `DEPLOYMENT_CONFIG_FILE`.

    If an image is not provided it will use the 'monetdb_image' field from
    the 'DEPLOYMENT_CONFIG_FILE' ex. monetdb_image = "madgik/mipenginedb:dev1.2"

    The data of the monetdb container are not persisted. If the container is recreated, all data will be lost.
    """
    if not node:
        message("Please specify a node using --node <node>", Level.WARNING)
        sys.exit(1)

    if not monetdb_image:
        monetdb_image = get_deployment_config("monetdb_image")

    get_docker_image(c, monetdb_image)

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
        cmd = f"docker run -d -P -p {container_ports} --name {container_name} {monetdb_image}"
        run(c, cmd)


@task(iterable=["port"])
def load_data(c, port=None):
    """
    Load data into the specified DB from the 'DEMO_DATA_FOLDER'.

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
            if "local" in node_config["identifier"]:
                local_node_ports.append(node_config["monetdb"]["port"])

    with open(NODE_CONFIG_TEMPLATE_FILE) as fp:
        template_node_config = toml.load(fp)
    for port in local_node_ports:
        message(f"Loading data on MonetDB at port {port}...", Level.HEADER)
        cmd = (
            f"poetry run python -m mipengine.node.monetdb_interface.csv_importer "
            f"-folder {DEMO_DATA_FOLDER} "
            f"-user {template_node_config['monetdb']['username']} "
            f"-pass {template_node_config['monetdb']['password']} "
            f"-url localhost:{port} "
            f"-farm db"
        )
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
    celery_log_level=None,
    detached=False,
    algorithm_folders=None,
):
    """
    (Re)Start the node(s) service(s). If a node service is running, stop and start it again.

    :param node: The node to start, using the proper file in the `NODES_CONFIG_DIR`.
    :param all_: If set, the nodes of which the configuration file exists, will be started.
    :param celery_log_level: If not provided, it will look into the `DEPLOYMENT_CONFIG_FILE`.
    :param detached: If set to True, it will start the service in the background.
    :param algorithm_folders: Used from the services. If not provided, it looks in the `DEPLOYMENT_CONFIG_FILE`.

    The containers related to the services remain unchanged.
    """

    if not celery_log_level:
        celery_log_level = get_deployment_config("celery_log_level")

    if not algorithm_folders:
        algorithm_folders = get_deployment_config("algorithm_folders")

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
                        f"-A mipengine.node.node worker -l {celery_log_level} >> {outpath} "
                        f"--purge 2>&1"
                    )
                    run(c, cmd, wait=False)
                else:
                    cmd = (
                        f"PYTHONPATH={PROJECT_ROOT} poetry run celery -A "
                        f"mipengine.node.node worker -l {celery_log_level} --purge"
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
    celery_log_level=None,
    monetdb_image=None,
    algorithm_folders=None,
):
    """
    Install dependencies, (re)create all the containers and (re)start all the services.

    :param install_dep: Install dependencies or not.
    :param start_all: Start all node/controller services flag.
    :param start_controller_: Start controller services flag.
    :param start_nodes: Start all nodes flag.
    :param celery_log_level: Used for the engine services. If not provided, it looks in the `DEPLOYMENT_CONFIG_FILE`.
    :param monetdb_image: Used for the db containers. If not provided, it looks in the `DEPLOYMENT_CONFIG_FILE`.
    :param algorithm_folders: Used from the services.
    """

    if not celery_log_level:
        celery_log_level = get_deployment_config("celery_log_level")

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
    for node_config_file in config_files:
        with open(node_config_file) as fp:
            node_config = toml.load(fp)
        node_ids.append(node_config["identifier"])

    create_monetdb(c, node=node_ids, monetdb_image=monetdb_image)
    create_rabbitmq(c, node=node_ids)

    if start_nodes or start_all:
        start_node(
            c,
            all_=True,
            celery_log_level=celery_log_level,
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


def run(c, cmd, attach_=False, wait=True, warn=False, raise_error=False, show_ok=True):
    if attach_:
        c.run(cmd, pty=True)
        return

    if not wait:
        # TODO disown=True will make c.run(..) return immediatelly
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
    result = run(c, cmd, show_ok=False)
    if result.stdout != "":
        return

    message(f"Pulling image {image} ...", Level.HEADER)
    cmd = f"docker pull {image}"
    run(c, cmd)

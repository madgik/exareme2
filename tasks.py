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

PROJECT_ROOT = Path(__file__).parent
DEPLOYMENT_CONFIG_FILE = PROJECT_ROOT / ".deployment.toml"
NODES_CONFIG_DIR = PROJECT_ROOT / "configs" / "nodes"
NODE_CONFIG_TEMPLATE_FILE = PROJECT_ROOT / "mipengine" / "node" / "config.toml"
OUTDIR = Path("/tmp/mipengine/")
if not OUTDIR.exists():
    OUTDIR.mkdir()

CONSUL_AGENT_CONTAINER_NAME = "consul-agent"

# TODO Add pre-tasks when this is implemented https://github.com/pyinvoke/invoke/issues/170
# Right now if we call a task from another task, the "pre"-task is not executed


@task
def create_node_configs(c):
    """
    This command, using the .deployment.toml file, will create the node configuration files.
    """

    if not Path(DEPLOYMENT_CONFIG_FILE).is_file():
        raise FileNotFoundError("Deployment config file '.deployment.toml' not found.")

    with open(DEPLOYMENT_CONFIG_FILE) as fp:
        deployment_config = toml.load(fp)

    with open(NODE_CONFIG_TEMPLATE_FILE) as fp:
        template_node_config = toml.load(fp)

    for node in deployment_config["nodes"]:
        node_config = template_node_config.copy()

        node_config["node_registry"]["ip"] = deployment_config["ip"]
        node_config["node_registry"]["port"] = deployment_config["node_registry_port"]

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


@task
def install_dependencies(c):
    """Install project dependencies using poetry"""
    message("Installing dependencies...", Level.HEADER)
    cmd = "poetry install"
    run(c, cmd)


@task
def rm_containers(c, container_name=None, monetdb=False, rabbitmq=False):
    """Remove containers

    Removes all containers having either monetdb or rabbitmq in the name."""
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


@task
def start_node_registry(context, container_name=None, port=None):
    if not container_name:
        container_name = CONSUL_AGENT_CONTAINER_NAME
    if not port:
        port = get_deployment_config("node_registry_port")

    # TODO killing the existing container is not obvious from the task name
    kill_node_registry(context)

    message(
        f"Starting container {container_name} on port {port}...",
        Level.HEADER,
    )
    # start the consul container
    cmd = f"docker run -d --name={container_name}  -p {port}:8500 consul"
    try:
        run(context, cmd, raise_error=True)
    # TODO this does not catch all exceptions, I think due to the async in the run function
    except (UnexpectedExit, AttributeError) as exc:
        print(f"{exc=}")


@task
def kill_node_registry(context, container_name=None):
    if not container_name:
        container_name = CONSUL_AGENT_CONTAINER_NAME

    # remove the consul container
    rm_containers(context, container_name=container_name)


@task(iterable=["node"])
def start_monetdb(c, node, monetdb_image=None):
    """Start MonetDB container(s) of given node(s)"""
    if not node:
        message("Please specify a node using --node <node>", Level.WARNING)
        sys.exit(1)

    if not monetdb_image:
        monetdb_image = get_deployment_config("monetdb_image")

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
    """Load data into DB from csv

    If the port is not set, the configurations inside the `./configs/nodes` folder
    will be used to load the data in the nodes. The data will be imported on nodes
    that have the `local` keyword in their name."""

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

    from tests import integration_tests

    data_folder = Path(integration_tests.__file__).parent / "data"
    with open(NODE_CONFIG_TEMPLATE_FILE) as fp:
        template_node_config = toml.load(fp)
    for port in local_node_ports:
        message(f"Loading data on MonetDB at port {port}...", Level.HEADER)
        cmd = (
            f"poetry run python -m mipengine.node.monetdb_interface.csv_importer "
            f"-folder {data_folder} "
            f"-user {template_node_config['monetdb']['username']} "
            f"-pass {template_node_config['monetdb']['password']} "
            f"-url localhost:{port} "
            f"-farm db"
        )
        run(c, cmd)


@task(iterable=["node"])
def config_rabbitmq(c, node):
    """Configure users and permissions for RabbitMQ containers of given node(s)"""
    message("Configuring RabbitMQ containers, this may take some time", Level.HEADER)

    with open(NODE_CONFIG_TEMPLATE_FILE) as fp:
        node_config = toml.load(fp)
    rabbitmqctl_cmds = [
        f"add_user {node_config['rabbitmq']['user']} {node_config['rabbitmq']['password']}",
        f"add_vhost {node_config['rabbitmq']['vhost']}",
        f"set_user_tags {node_config['rabbitmq']['user']} user_tag",
        f"set_permissions -p {node_config['rabbitmq']['vhost']} {node_config['rabbitmq']['user']} '.*' '.*' '.*'",
    ]
    node_ids = node
    for node_id in node_ids:
        container_name = f"rabbitmq-{node_id}"

        for rmq_cmd in rabbitmqctl_cmds:
            message(
                f"Configuring container {container_name}: rabbitmqctl {rmq_cmd}...",
                Level.HEADER,
            )
            cmd = f"docker exec {container_name} rabbitmqctl {rmq_cmd}"
            for _ in range(30):
                try:
                    run(c, cmd, raise_error=True)
                except UnexpectedExit:
                    spin_wheel(time=2)
                else:
                    break
            else:
                message("Cannot configure RabbitMQ", Level.ERROR)
                sys.exit(1)


@task(iterable=["node"])
def start_rabbitmq(c, node):
    """Start RabbitMQ container(s) of given node(s)"""
    if not node:
        message("Please specify a node using --node <node>", Level.WARNING)
        sys.exit(1)

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
        cmd = f"docker run -d -p {container_ports} --name {container_name} rabbitmq"
        run(c, cmd)


@task
def kill_node(c, node=None, all_=False):
    """Kill Celery node

    The method always tries two commands, one for cases where node was
    started using the celery binary and one for cases it was started
    as a python module.

    In order for the node processes to be killed, we need to kill both
    the parent process with the 'node_identifier' and it's child."""

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
        cmd = (
            f"pid=$(ps aux | grep '[c]elery' | grep 'worker' | grep '{node_pattern}' | awk '{{print $2}}') "
            f"&& pgrep -P $pid | xargs kill -9 "
            f"&& kill -9 $pid "
        )
        run(c, cmd, warn=True)
    else:
        message("No celery instances found", Level.HEADER)


@task
def start_node(c, node=None, all_=False, celery_log_level=None, detached=False):
    """Start Celery node(s)

    A node is started using the appropriate file inside the ./configs/nodes folder.
    A file with the same name as the node should exist.

    If the --all argument is given, the nodes of which the configuration file exists, will be started."""

    if not celery_log_level:
        celery_log_level = get_deployment_config("celery_log_level")

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

    for node_id in node_ids:
        kill_node(c, node_id)

        message(f"Starting Node {node_id}...", Level.HEADER)
        node_config_file = NODES_CONFIG_DIR / f"{node_id}.toml"
        with c.prefix(f"export MIPENGINE_NODE_CONFIG_FILE={node_config_file}"):
            outpath = OUTDIR / (node_id + ".out")
            if detached or all_:
                cmd = (
                    f"PYTHONPATH={PROJECT_ROOT} poetry run celery "
                    f"-A mipengine.node.node worker -l {celery_log_level} >> {outpath} 2>&1"
                )
                run(c, cmd, wait=False)
            else:
                cmd = f"PYTHONPATH={PROJECT_ROOT} poetry run celery -A mipengine.node.node worker -l {celery_log_level}"
                run(c, cmd, attach_=True)


@task
def kill_controller(c):
    """Kill Controller"""
    res = run(c, "ps aux | grep '[q]uart'", warn=True, show_ok=False)
    if res.ok:
        message("Killing previous Quart instances...", Level.HEADER)
        cmd = "ps aux | grep '[q]uart' | awk '{ print $2}' | xargs kill -9 && sleep 5"
        run(c, cmd)
    else:
        message("No quart instance found", Level.HEADER)


@task
def start_controller(c, detached=False):
    """Start Controller"""
    kill_controller(c)

    message("Starting Controller...", Level.HEADER)
    with c.prefix("export QUART_APP=mipengine/controller/api/app:app"):
        outpath = OUTDIR / "controller.out"
        if detached:
            cmd = f"PYTHONPATH={PROJECT_ROOT} poetry run quart run >> {outpath} 2>&1"
            run(c, cmd, wait=False)
        else:
            cmd = f"PYTHONPATH={PROJECT_ROOT} poetry run quart run"
            run(c, cmd, attach_=True)


@task
def deploy(
    c,
    install_dep=True,
    start_all=False,
    start_controller_=False,
    start_nodes=False,
    celery_log_level=None,
    monetdb_image=None,
):
    """(Re)Deploy everything.
    The nodes will be deployed using the existing node config files."""

    if not celery_log_level:
        celery_log_level = get_deployment_config("celery_log_level")

    if not monetdb_image:
        monetdb_image = get_deployment_config("monetdb_image")

    if install_dep:
        install_dependencies(c)

    #start NODE SERVICE service
    start_node_registry(c)

    #start NODE services
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

    start_monetdb(c, node=node_ids, monetdb_image=monetdb_image)
    start_rabbitmq(c, node=node_ids)
    config_rabbitmq(c, node=node_ids)

    if start_nodes or start_all:
        start_node(c, all_=True, celery_log_level=celery_log_level, detached=True)

    #start CONTROLLER service
    if start_controller_ or start_all:
        start_controller(c, detached=True)

@task
def attach(c, node=None, controller=False, db=None):
    """Attach to Node, Controller or DB"""
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
    """Kill Controller and Nodes, remove MonetDB and RabbitMQ containers"""
    kill_controller(c)
    kill_node(c, all_=True)
    rm_containers(c, monetdb=True, rabbitmq=True)
    kill_node_registry(c)
    if OUTDIR.exists():
        message(f"Removing {OUTDIR}...", level=Level.HEADER)
        for outpath in OUTDIR.glob("*.out"):
            outpath.unlink()
        OUTDIR.rmdir()
        message("Ok", level=Level.SUCCESS)


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
    # TODO this is also obscure. := makes it obscure
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
            f"Please provide a --{config} parameter or create a deployment config file '.deployment.toml'"
        )

    with open(DEPLOYMENT_CONFIG_FILE) as fp:
        return toml.load(fp)[config]

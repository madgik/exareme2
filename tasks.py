import sys
from enum import Enum
from itertools import cycle
from pathlib import Path
from textwrap import indent
from time import sleep

import toml
from invoke import UnexpectedExit
from invoke import call
from invoke import task
from termcolor import colored

PROJECT_ROOT = Path(__file__).parent
ENVFILE = PROJECT_ROOT / ".mipenv"
CONFIG_FILES_FOLDER = PROJECT_ROOT / "configs/"
DEFAULT_NODE_CONFIG_FILE = PROJECT_ROOT / "mipengine/node/config.toml"
OUTDIR = Path("/tmp/mipengine/")
if not OUTDIR.exists():
    OUTDIR.mkdir()


@task(iterable=["node_name", "monetdb_port", "rabbitmq_port"])
def config(
    c, ip=None, node_name=None, monetdb_port=None, rabbitmq_port=None, show=False
):
    """Configure mipengine deployment

    Run this command to configure the mipengine deployment process."""
    if ip and node_name and monetdb_port and rabbitmq_port:
        if not ENVFILE.exists():
            ENVFILE.touch()
        if not (len(node_name) == len(monetdb_port) == len(rabbitmq_port)):
            message(
                "You should provide an equal number of node names, monetdb ports and rabbitmq ports",
                Level.ERROR,
            )
            sys.exit(1)
        envvars = [
            f"export PYTHONPATH={PROJECT_ROOT}",
            f"export MIPENGINE_LOCAL_IP={ip}",
            f"export MIPENGINE_NODE_NAMES={':'.join(node_name)}",
            f"export MIPENGINE_MONETDB_PORTS={':'.join(monetdb_port)}",
            f"export MIPENGINE_RABBITMQ_PORTS={':'.join(rabbitmq_port)}",
        ]
        envvars = [line + "\n" for line in envvars]
        with ENVFILE.open("w") as f:
            f.writelines(envvars)
        print_config()
    elif show:
        if not ENVFILE.exists():
            message("No config found, run invoke config --help", level=Level.WARNING)
        else:
            print_config()
    else:
        message(
            "You must either specify all config parameters or use --show flag to show current config",
            level=Level.WARNING,
        )


@task
def install(c):
    """Install project dependencies using poetry"""
    message("Installing dependencies...", Level.HEADER)
    cmd = "poetry install"
    run(c, cmd)


@task
def set_ip(c, ip):
    """Configure mipengine with you machine's IP"""
    message("Setting ip address...", Level.HEADER)
    cmd = f"poetry run python tests/integration_tests/set_hostname_in_node_catalog.py -host {ip}"
    run(c, cmd)


@task
def rm_containers(c, monetdb=False, rabbitmq=False):
    """Remove containers

    Removes all containers having either monetdb or rabbitmq in the name."""
    names = []
    if monetdb:
        names.append("monetdb")
    if rabbitmq:
        names.append("rabbitmq")
    if not names:
        message(
            "You must specify at least one container family to remove (monetdb or/and rabbitmq)",
            level=Level.WARNING,
        )
    for name in names:
        container_ids = c.run(f"docker ps -q --filter name={name}", hide="out")
        if container_ids.stdout:
            message(f"Removing {name} containers...", Level.HEADER)
            cmd = f"docker rm -vf $(docker ps -q --filter name={name})"
            run(c, cmd)
        else:
            message(f"No {name} container to remove", level=Level.HEADER)


@task(pre=[call(rm_containers, monetdb=True)], iterable=["port"])
def start_monetdb(c, port):
    """Start MonetDB container(s) on given port(s)"""
    ports = port
    for i, port in enumerate(ports):
        container_ports = f"{port}:50000"
        container_name = f"monetdb-{i}"
        message(
            f"Starting container {container_name} on ports {container_ports}...",
            Level.HEADER,
        )
        cmd = f"docker run -d -P -p {container_ports} --name {container_name} jassak/mipenginedb:dev1.1"
        run(c, cmd)


@task(iterable=["port"])
def load_data_into_db(c, port):
    """Load data into DB from csv"""
    ports = port
    for port in ports:
        message(f"Loading data on MonetDB at port {port}...", Level.HEADER)
        cmd = f"poetry run python -m mipengine.node.monetdb_interface.csv_importer -folder ./tests/data/ -user monetdb -pass monetdb -url localhost:{port} -farm db"
        run(c, cmd)


@task
def config_rabbitmq(c, ports):
    """Configure users and permissions for RabbitMQ containers"""
    message("Password required for configuring RabbitMQ containers:", Level.WARNING)
    c.run("sudo echo")
    message("Configuring RabbitMQ containers, this may take some time", Level.HEADER)
    rabbitmqctl_cmds = [
        "add_user user password",
        "add_vhost user_vhost",
        "set_user_tags user user_tag",
        "set_permissions -p user_vhost user '.*' '.*' '.*'",
    ]
    for num, port in enumerate(ports):
        container_name = f"rabbitmq-{num}"

        for rmq_cmd in rabbitmqctl_cmds:
            message(
                f"Configuring container {container_name}: rabbitmqctl {rmq_cmd}...",
                Level.HEADER,
            )
            cmd = f"sudo docker exec {container_name} rabbitmqctl {rmq_cmd}"
            for _ in range(30):
                try:
                    # only works with c.run for some crazy reason
                    c.run(cmd, hide="both")
                except UnexpectedExit as err:
                    if err.result.return_code in (69, 64):
                        sleep(2)
                    else:
                        message("Error", Level.ERROR)
                        message(err.result.stderr, Level.BODY)
                        sys.exit(err.result.return_code)
                else:
                    message("Ok", Level.SUCCESS)
                    break
            else:
                message("Cannot configure RabbitMQ", Level.ERROR)
                sys.exit(1)


@task(pre=[call(rm_containers, rabbitmq=True)], iterable=["port"])
def start_rabbitmq(c, port):
    """Start RabbitMQ container(s) on given port(s)"""
    ports = port
    for i, port in enumerate(ports):
        container_name = f"rabbitmq-{i}"
        container_ports = f"{port}:5672"
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
    as a python module."""
    if all_:
        node_pattern = ""
    elif node:
        node_pattern = node
    else:
        message("Please specify a node using --node <node> or use --all", Level.WARNING)
        sys.exit(1)
    node_descr = f" {node_pattern}" if node_pattern else "s"
    res_bin = c.run(
        f"ps aux | grep '[c]elery' | grep 'worker' | grep '{node_pattern}' ",
        hide="both",
        warn=True,
    )
    res_py = c.run(
        f"ps aux | grep '[m]ipengine' | grep 'worker' | grep '{node_pattern}'",
        hide="both",
        warn=True,
    )
    if res_bin.ok:
        message(
            f"Killing previous celery instance{node_descr} started using celery binary...",
            Level.HEADER,
        )
        cmd = f"ps aux | grep '[c]elery' | grep 'worker' | grep '{node_pattern}' | awk '{{print $2}}' | xargs kill -9"
        c.run(cmd)
    if res_py.ok:
        message(
            f"Killing previous celery instance{node_descr} started as a python module...",
            Level.HEADER,
        )
        cmd = f"ps aux | grep '[m]ipengine' | grep 'worker' | grep '{node_pattern}' | awk '{{print $2}}' | xargs kill -9"
        run(c, cmd)
    if not res_bin.ok and not res_py.ok:
        message("No celery instances found", Level.HEADER)


@task
def attach(c, node=None, controller=False, db=None):
    """Attach to Node, Controller or DB"""
    if (node or controller) and not (node and controller):
        fname = node or "controller"
        outpath = OUTDIR / (fname + ".out")
        cmd = f"tail -f {outpath}"
        c.run(cmd)
    elif db:
        c.run(f"docker exec -it {db} mclient db", pty=True)
    else:
        message("You must attach to Node, Controller or DB", Level.WARNING)
        sys.exit(1)


@task(iterable=["node_name", "monetdb_port", "rabbitmq_port"])
def start_node(c, ip, node_name, monetdb_port, rabbitmq_port):
    """Start Celery node(s)"""
    node_names = node_name
    monetdb_ports = monetdb_port
    rabbitmq_ports = rabbitmq_port

    if ip and node_names and monetdb_ports and rabbitmq_ports:
        if not (len(node_names) == len(monetdb_ports) == len(rabbitmq_ports)):
            message(
                "You should provide an equal number of node names, monetdb ports and rabbitmq ports",
                Level.ERROR,
            )
            sys.exit(1)
    else:
        message(
            "You must specify all of the ip, node, monetdb port and rabbitmq port parameters.",
            level=Level.ERROR,
        )
        sys.exit(1)

    for (node_name, monetdb_port, rabbitmq_port) in zip(
        node_names, monetdb_ports, rabbitmq_ports
    ):
        kill_node(c, node_name)
        message(f"Starting Node {node_name}...", Level.HEADER)

        config_file = create_node_config_file(
            node_name, ip, monetdb_port, rabbitmq_port
        )
        c.prefix(f"export CONFIG_FILE={config_file}")
        outpath = OUTDIR / (node_name + ".out")
        cmd = (
            f"poetry run python -m mipengine.node.node worker -l info >> {outpath} 2>&1"
        )
        c.run(cmd, disown=True)
        spin_wheel(time=4)
        message("Ok", Level.SUCCESS)


@task
def kill_controller(c):
    """Kill Controller"""
    res = c.run("ps aux | grep '[q]uart'", hide="both", warn=True)
    if res.ok:
        message("Killing previous Quart instances...", Level.HEADER)
        cmd = "ps aux | grep '[q]uart' | awk '{ print $2}' | xargs kill -9 && sleep 5"
        run(c, cmd)
    else:
        message("No quart instance found", Level.HEADER)


@task(pre=[kill_controller])
def start_controller(c):
    """Start Controller"""
    message("Starting Controller...", Level.HEADER)
    with c.prefix("export QUART_APP=mipengine/controller/api/app:app"):
        outpath = OUTDIR / "controller.out"
        cmd = f"poetry run quart run >> {outpath} 2>&1"
        c.run(cmd, disown=True)
    spin_wheel(time=4)
    message("Ok", Level.SUCCESS)


@task
def deploy(c, start_controller_=False, start_nodes=False, install_=True):
    """Deploy everything"""
    with c.cd(PROJECT_ROOT):
        if not ENVFILE.exists():
            message(
                "No config found, run invoke config to create one",
                Level.WARNING,
            )
            sys.exit(1)
        with c.prefix(f"source {ENVFILE}"):
            message("Using the following config", Level.HEADER)
            print_config()
            local_ip = c.run("echo $MIPENGINE_LOCAL_IP", hide="out").stdout.strip("\n")
            monetdb_ports = (
                c.run("echo $MIPENGINE_MONETDB_PORTS", hide="out")
                .stdout.strip("\n")
                .split(":")
            )
            rabbitmq_ports = (
                c.run("echo $MIPENGINE_RABBITMQ_PORTS", hide="out")
                .stdout.strip("\n")
                .split(":")
            )
            node_names = (
                c.run("echo $MIPENGINE_NODE_NAMES", hide="out")
                .stdout.strip("\n")
                .split(":")
            )

            if install_:
                install(c)
            set_ip(c, local_ip)

            rm_containers(c, monetdb=True)
            start_monetdb(c, monetdb_ports)

            rm_containers(c, rabbitmq=True)
            start_rabbitmq(c, rabbitmq_ports)
            config_rabbitmq(c, rabbitmq_ports)

            if start_controller_:
                kill_controller(c)
                start_controller(c)
            if start_nodes:
                start_node(
                    c,
                    node_name=node_names,
                    monetdb_port=monetdb_ports,
                    rabbitmq_port=rabbitmq_ports,
                )


@task
def cleanup(c):
    """Kill Controller and Nodes, remove MonetDB and RabbitMQ containers"""
    kill_controller(c)
    kill_node(c, all_=True)
    rm_containers(c, monetdb=True, rabbitmq=True)
    if OUTDIR.exists():
        message(f"Removing {OUTDIR}...", level=Level.HEADER)
        for outpath in OUTDIR.glob("*.out"):
            outpath.unlink()
        OUTDIR.rmdir()
        message("Ok", level=Level.SUCCESS)


def run(c, cmd, finish=True, error_check=True):
    promise = c.run(cmd, asynchronous=True)
    spin_wheel(promise=promise)
    stderr = promise.runner.stderr
    if error_check and stderr:
        message("Error", Level.ERROR)
        message("\n".join(stderr), Level.BODY)
        sys.exit(promise.runner.returncode())
    elif finish:
        message("Ok", Level.SUCCESS)


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


def print_config():
    message("\nMIP config:", Level.BODY)
    message("========== ", Level.BODY)
    if ENVFILE.exists:
        with ENVFILE.open("r") as f:
            for line in f.readlines():
                message(line.lstrip("export ").rstrip("\n"), Level.BODY)
    print()


def create_node_config_file(node_id, ip, monetdb_port, rabbitmq_port):
    with open(DEFAULT_NODE_CONFIG_FILE) as fp:
        node_config = toml.load(fp)

    node_config["monetdb"]["ip"] = ip
    node_config["monetdb"]["port"] = monetdb_port
    node_config["rabbitmq"]["ip"] = ip
    node_config["rabbitmq"]["port"] = rabbitmq_port

    Path(CONFIG_FILES_FOLDER).mkdir(parents=True, exist_ok=True)
    node_config_file = CONFIG_FILES_FOLDER / f"{node_id}.toml"
    with open(node_config_file, "w+") as fp:
        toml.dump(node_config, fp)

    return node_config_file

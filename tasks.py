from time import sleep
from enum import Enum

from invoke import task, UnexpectedExit
from termcolor import colored


@task
def install_dependencies(c):
    message("Installing dependencies...", Level.HEADER)
    cmd = "poetry install"
    message(cmd)
    c.run(cmd, hide="out")
    message("Done!", Level.SUCCESS)


@task
def set_ip(c, ip):
    message("Setting ip address...", Level.HEADER)
    with c.cd("mipengine/tests/node"):
        cmd = f"poetry run python set_hostname_in_node_catalog.py -host {ip}"
        message(cmd)
        c.run(cmd, hide="out")
    message("Done!", Level.SUCCESS)


@task
def clean_monetdb_containers(c):
    message("Cleaning up MonetDB containers...", Level.HEADER)
    cmd = "docker ps -a | grep -E 'monetdb' | awk '{ print $1 }' | xargs docker rm -vf"
    message(cmd)
    c.run(cmd, hide="out")
    message("Done!", Level.SUCCESS)


@task
def start_monetdb_global(c):
    message("Starting Global Node MonetDB instance...", Level.HEADER)
    cmd = "docker run -d -P -p 50000:50000 --name monetdb-0 jassak/mipenginedb:dev1.1"
    message(cmd)
    c.run(cmd, hide="out")
    message("Done!", Level.SUCCESS)


@task
def start_monetdb_locals(c, local_nodes=2):
    message(f"Starting {local_nodes} Local Node MonetDB instances...", Level.HEADER)
    for i in range(1, local_nodes + 1):
        cmd = f"docker run -d -P -p 5000{i}:50000 --name monetdb-{i} jassak/mipenginedb:dev1.1"
        message(cmd)
        c.run(cmd, hide="out")
    message("Done!", Level.SUCCESS)


@task
def load_data_into_db(c, local_nodes=2):
    message("Loading data...", Level.HEADER)
    for i in range(1, local_nodes + 1):
        cmd = f"poetry run python -m mipengine.node.monetdb_interface.csv_importer -folder ./mipengine/tests/data/ -user monetdb -pass monetdb -url localhost:5000{i} -farm db"
        message(cmd)
        c.run(cmd, hide="out")
    message("Done!", Level.SUCCESS)


@task
def setup_dbs(c, local_nodes=2):
    message("Deploying MonetDB in all nodes...\n", Level.HEADER)
    clean_monetdb_containers(c)
    start_monetdb_global(c)
    start_monetdb_locals(c, local_nodes=local_nodes)
    load_data_into_db(c, local_nodes=local_nodes)


@task
def clean_rabbitmq_containers(c):
    message("Cleaning up RabbitMQ containers...", Level.HEADER)
    cmd = "docker ps -a | grep -E 'rabbitmq' | awk '{ print $1 }' | xargs docker rm -vf"
    message(cmd)
    c.run(cmd, hide="out")
    message("Done!", Level.SUCCESS)


@task
def config_rabbitmq(c):
    message("Configuring RabbitMQ...", Level.HEADER)
    cmd = "docker exec rabbitmq-0 rabbitmqctl add_user user password"
    for _ in range(30):
        try:
            c.run(cmd, hide="both")
        except UnexpectedExit as err:
            sleep(2)
        else:
            break
    else:
        message("Cannot configure RabbitMQ. Exiting.", Level.ERROR)
        return
    message(cmd)

    cmd = "docker exec rabbitmq-0 rabbitmqctl add_vhost user_vhost"
    message(cmd)
    c.run(cmd, hide="out")
    cmd = "docker exec rabbitmq-0 rabbitmqctl set_user_tags user user_tag"
    message(cmd)
    c.run(cmd, hide="out")
    cmd = "docker exec rabbitmq-0 rabbitmqctl set_permissions -p user_vhost user '.*' '.*' '.*'"
    message(cmd)
    c.run(cmd, hide="out")

    cmd = "docker exec rabbitmq-1 rabbitmqctl add_user user password"
    message(cmd)
    c.run(cmd, hide="out")
    cmd = "docker exec rabbitmq-1 rabbitmqctl add_vhost user_vhost"
    message(cmd)
    c.run(cmd, hide="out")
    cmd = "docker exec rabbitmq-1 rabbitmqctl set_user_tags user user_tag"
    message(cmd)
    c.run(cmd, hide="out")
    cmd = "docker exec rabbitmq-1 rabbitmqctl set_permissions -p user_vhost user '.*' '.*' '.*'"
    message(cmd)
    c.run(cmd, hide="out")

    cmd = "docker exec rabbitmq-2 rabbitmqctl add_user user password"
    message(cmd)
    c.run(cmd, hide="out")
    cmd = "docker exec rabbitmq-2 rabbitmqctl add_vhost user_vhost"
    message(cmd)
    c.run(cmd, hide="out")
    cmd = "docker exec rabbitmq-2 rabbitmqctl set_user_tags user user_tag"
    message(cmd)
    c.run(cmd, hide="out")
    cmd = "docker exec rabbitmq-2 rabbitmqctl set_permissions -p user_vhost user '.*' '.*' '.*'"
    message(cmd)
    c.run(cmd, hide="out")
    message("Done!", Level.SUCCESS)


@task(pre=[clean_rabbitmq_containers], post=[config_rabbitmq])
def start_rabbitmq(c):
    message("Starting RabbitMQ instances...", Level.HEADER)
    cmd = "docker run -d -p 5670:5672 --name rabbitmq-0 rabbitmq"
    message(cmd)
    c.run(cmd, hide="out")
    cmd = "docker run -d -p 5671:5672 --name rabbitmq-1 rabbitmq"
    message(cmd)
    c.run(cmd, hide="out")
    cmd = "docker run -d -p 5672:5672 --name rabbitmq-2 rabbitmq"
    message(cmd)
    c.run(cmd, hide="out")
    message("Done!", Level.SUCCESS)


@task
def setup_rabbitmq(c):
    clean_rabbitmq_containers(c)
    start_rabbitmq(c)
    config_rabbitmq(c)


@task
def killall_celery(c):
    message("Killing previous Celery instances...", Level.HEADER)
    cmd = "ps aux | grep '[c]elery' | awk '{ print $2}' | xargs kill -9"
    message(cmd)
    c.run(cmd)
    message("Done!", Level.SUCCESS)


@task(pre=[killall_celery])
def start_global_node(c):
    message("Starting Global Node...", Level.HEADER)
    cmd = "poetry run python mipengine/tests/node/set_node_identifier.py globalnode && poetry run celery -A mipengine.node.node worker -l INFO"
    message(cmd)
    c.run(cmd, disown=True)
    sleep(4)
    message("Done!", Level.SUCCESS)


@task(pre=[killall_celery])
def start_local_nodes(c, local_nodes=2):
    message(f"Starting {local_nodes} Local Nodes...", Level.HEADER)
    for i in range(1, local_nodes + 1):
        cmd = f"poetry run python mipengine/tests/node/set_node_identifier.py localnode{i} && poetry run celery -A mipengine.node.node worker -l INFO"
        message(cmd)
        c.run(cmd, disown=True)
    sleep(4)
    message("Done!", Level.SUCCESS)


@task
def killall_quart(c):
    message("Killing previous Quart instances...", Level.HEADER)
    cmd = "ps aux | grep '[q]uart' | awk '{ print $2}' | xargs kill -9"
    message(cmd)
    c.run(cmd)
    message("Done!", Level.SUCCESS)


@task(pre=[killall_quart])
def start_controller(c):
    message("Starting Controller...", Level.HEADER)
    with c.prefix("export QUART_APP=mipengine/controller/api/app:app"):
        cmd = "poetry run quart run"
        c.run(cmd, disown=True)
    sleep(4)
    message("Done!", Level.SUCCESS)


@task(
    optional=["start_services"],
)
def deploy(c, start_services=False, ip=None):
    if not ip:
        message(
            "Please provide you IP with the --ip flag (see invoke deploy --help)",
            Level.ERROR,
        )
        return
    install_dependencies(c)
    set_ip(c, ip)
    setup_dbs(c)
    setup_rabbitmq(c)
    if start_services:
        start_controller(c)
        start_global_node(c)
        start_local_nodes(c)
    message("Deployed everything", Level.SUCCESS)


class Level(Enum):
    HEADER = ("cyan", 1)
    BODY = ("white", 2)
    SUCCESS = ("green", 1)
    ERROR = ("red", 1)


def message(msg, level=Level.BODY):
    end = "\n\n" if level is Level.SUCCESS else "\n"
    color, indent = level.value
    msg = colored(msg, color)
    print(" " * 2 * indent + msg, end=end, flush=True)

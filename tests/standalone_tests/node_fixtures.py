import pytest
import docker
import time
import subprocess
import os
from os import path
import toml

from mipengine.controller.node_tasks_handler_celery import NodeTasksHandlerCelery


TESTING_RABBITMQ_CONT_IMAGE = "madgik/mipengine_rabbitmq:latest"
TESTING_RABBITMQ_CONT_NAME = "rabbitmq-testlocalnode1"
TESTING_RABBITMQ_CONT_PORT = "5700"

this_mod_path = os.path.dirname(os.path.abspath(__file__))
TEST_NODE_CONFIG_FILE = path.join(
    path.join(this_mod_path, "testing_env_configs"), "testnode1.toml"
)
TASKS_TIMEOUT = 5


@pytest.fixture(autouse=True, scope="function")
def rabbitmq_container():
    client = docker.from_env()
    try:
        container = client.containers.get(TESTING_RABBITMQ_CONT_NAME)
    except docker.errors.NotFound:
        container = client.containers.run(
            TESTING_RABBITMQ_CONT_IMAGE,
            detach=True,
            ports={"5672/tcp": TESTING_RABBITMQ_CONT_PORT},
            name=TESTING_RABBITMQ_CONT_NAME,
        )

    while (
        "Health" not in container.attrs["State"]
        or container.attrs["State"]["Health"]["Status"] != "healthy"
    ):
        container.reload()  # attributes are cached, this refreshes them..
        time.sleep(1)

    ids = {"rabbitmq_container_id": container.id}
    yield ids
    try:
        container = client.containers.get(TESTING_RABBITMQ_CONT_NAME)
        container.remove(v=True, force=True)
    except docker.errors.NotFound:
        pass  # container was removed by other means..


@pytest.fixture(autouse=True, scope="function")
def node(rabbitmq_container, use_database):
    os.environ["ALGORITHM_FOLDERS_ENV_VARIABLE"] = "dummypath"
    os.environ["MIPENGINE_NODE_CONFIG_FILE"] = TEST_NODE_CONFIG_FILE

    cmd = f"poetry run celery -A mipengine.node.node worker -l INFO --purge "

    # if executed without "exec" it is spawned as a child process of the shell and it is
    # difficult to kill it
    proc = subprocess.Popen(
        "exec " + cmd, shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE
    )

    ids = rabbitmq_container
    ids["celery_app_pid"] = proc

    # the celery app needs sometime to be ready, we should have some kind of check
    # for that, for now just a sleep..
    time.sleep(10)
    yield ids
    proc.kill()


@pytest.fixture(scope="function")
def node_tasks_handler_celery(node):
    with open(TEST_NODE_CONFIG_FILE) as fp:
        tmp = toml.load(fp)
        node_id = tmp["identifier"]
        queue_domain = tmp["rabbitmq"]["ip"]
        queue_port = tmp["rabbitmq"]["port"]
        queue_address = ":".join([str(queue_domain), str(queue_port)])
        db_domain = tmp["monetdb"]["ip"]
        db_port = tmp["monetdb"]["port"]
        db_address = ":".join([str(db_domain), str(db_port)])
        tasks_timeout = TASKS_TIMEOUT
    return {
        "pids": node,
        "tasks_handler": NodeTasksHandlerCelery(
            node_id=node_id,
            node_queue_addr=queue_address,
            node_db_addr=db_address,
            tasks_timeout=tasks_timeout,
        ),
    }


def remove_rabbitmq_controller(container_id):
    cmd = f"docker rm -f {container_id}"
    subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)

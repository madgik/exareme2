import subprocess
import uuid

import fasteners
import pytest

from mipengine.node_info_DTOs import NodeInfo
from tests.dev_env_tests.nodes_communication import get_celery_app
from tests.dev_env_tests.nodes_communication import get_celery_task_signature


@pytest.fixture(autouse=True)
def run_tests_sequentially():
    with fasteners.InterProcessLock("semaphore.lock"):
        yield


test_cases_get_node_info = [
    (
        "globalnode",
        NodeInfo(
            id="globalnode",
            role="GLOBALNODE",
            ip="172.17.0.1",
            port=5670,
            db_ip="172.17.0.1",
            db_port=50000,
        ),
    ),
    (
        "localnode1",
        NodeInfo(
            id="localnode1",
            role="LOCALNODE",
            ip="172.17.0.1",
            port=5671,
            db_ip="172.17.0.1",
            db_port=50001,
        ),
    ),
]


@pytest.mark.parametrize(
    "node_id, expected_node_info",
    test_cases_get_node_info,
)
def test_get_node_info(node_id, expected_node_info):
    request_id = "test_node_info_" + uuid.uuid4().hex + "_request"
    node_app = get_celery_app(node_id)
    node_info_signature = get_celery_task_signature(node_app, "get_node_info")
    task_response = node_info_signature.delay(request_id=request_id).get()
    node_info = NodeInfo.parse_raw(task_response)

    assert node_info.id == expected_node_info.id
    assert node_info.role == expected_node_info.role

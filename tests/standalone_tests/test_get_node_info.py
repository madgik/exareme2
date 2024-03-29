import uuid

import pytest

from exareme2.worker_communication import WorkerInfo
from tests.standalone_tests.conftest import TASKS_TIMEOUT
from tests.standalone_tests.nodes_communication_helper import get_celery_task_signature
from tests.standalone_tests.std_output_logger import StdOutputLogger

node_info_per_node = {
    "testglobalworker": WorkerInfo(
        id="testglobalworker",
        role="GLOBALWORKER",
        ip="172.17.0.1",
        port=60000,
        db_ip="172.17.0.1",
        db_port=61000,
    ),
    "testlocalworker1": WorkerInfo(
        id="testlocalworker1",
        role="LOCALWORKER",
        ip="172.17.0.1",
        port=60001,
        db_ip="172.17.0.1",
        db_port=61001,
    ),
}


@pytest.mark.slow
def test_get_node_info(
    localworker1_worker_service,
    globalworker_worker_service,
    localworker1_celery_app,
    globalworker_celery_app,
):
    request_id = "test_node_info_" + uuid.uuid4().hex + "_request"

    task_signature = get_celery_task_signature("get_node_info")
    localnode1_async_result = localworker1_celery_app.queue_task(
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
    )
    localnode1_task_response = localworker1_celery_app.get_result(
        async_result=localnode1_async_result,
        logger=StdOutputLogger(),
        timeout=TASKS_TIMEOUT,
    )

    node_info = WorkerInfo.parse_raw(localnode1_task_response)

    assert node_info.id == node_info_per_node["testlocalworker1"].id
    assert node_info.role == node_info_per_node["testlocalworker1"].role

    globalnode_async_result = globalworker_celery_app.queue_task(
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
    )
    globalnode_task_response = globalworker_celery_app.get_result(
        async_result=globalnode_async_result,
        logger=StdOutputLogger(),
        timeout=TASKS_TIMEOUT,
    )

    node_info = WorkerInfo.parse_raw(globalnode_task_response)

    assert node_info.id == node_info_per_node["testglobalworker"].id
    assert node_info.role == node_info_per_node["testglobalworker"].role

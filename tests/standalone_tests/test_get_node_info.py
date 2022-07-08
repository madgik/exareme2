import uuid

from mipengine.node_info_DTOs import NodeInfo
from tests.standalone_tests.conftest import TASKS_TIMEOUT
from tests.standalone_tests.nodes_communication_helper import get_celery_task_signature
from tests.standalone_tests.std_output_logger import StdOutputLogger

node_info_per_node = {
    "testglobalnode": NodeInfo(
        id="testglobalnode",
        role="GLOBALNODE",
        ip="172.17.0.1",
        port=60000,
        db_ip="172.17.0.1",
        db_port=61000,
    ),
    "testlocalnode1": NodeInfo(
        id="testlocalnode1",
        role="LOCALNODE",
        ip="172.17.0.1",
        port=60001,
        db_ip="172.17.0.1",
        db_port=61001,
    ),
}


def test_get_node_info(
    localnode1_node_service,
    globalnode_node_service,
    localnode1_celery_app,
    globalnode_celery_app,
):
    request_id = "test_node_info_" + uuid.uuid4().hex + "_request"

    task_signature = get_celery_task_signature("get_node_info")
    localnode1_async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
    )
    localnode1_task_response = localnode1_celery_app.get_result(
        async_result=localnode1_async_result,
        logger=StdOutputLogger(),
        timeout=TASKS_TIMEOUT,
    )

    node_info = NodeInfo.parse_raw(localnode1_task_response)

    assert node_info.id == node_info_per_node["testlocalnode1"].id
    assert node_info.role == node_info_per_node["testlocalnode1"].role

    globalnode_async_result = globalnode_celery_app.queue_task(
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
    )
    globalnode_task_response = globalnode_celery_app.get_result(
        async_result=globalnode_async_result,
        logger=StdOutputLogger(),
        timeout=TASKS_TIMEOUT,
    )

    node_info = NodeInfo.parse_raw(globalnode_task_response)

    assert node_info.id == node_info_per_node["testglobalnode"].id
    assert node_info.role == node_info_per_node["testglobalnode"].role

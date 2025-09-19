import uuid

import pytest

from exareme2.worker_communication import MonetDBConfig
from exareme2.worker_communication import WorkerInfo
from tests.standalone_tests.conftest import TASKS_TIMEOUT
from tests.standalone_tests.controller.workers_communication_helper import (
    get_celery_task_signature,
)
from tests.standalone_tests.std_output_logger import StdOutputLogger

worker_info_per_worker = {
    "testglobalworker": WorkerInfo(
        id="testglobalworker",
        role="GLOBALWORKER",
        ip="172.17.0.1",
        port=60000,
        monetdb_configs=MonetDBConfig(port=61000, ip="172.17.0.1"),
    ),
    "testlocalworker1": WorkerInfo(
        id="testlocalworker1",
        role="LOCALWORKER",
        ip="172.17.0.1",
        port=60001,
        monetdb_configs=MonetDBConfig(port=61001, ip="172.17.0.1"),
    ),
}


@pytest.mark.slow
def test_get_worker_info(
    monetdb_localworker1,
    localworker1_worker_service,
    monetdb_globalworker,
    globalworker_worker_service,
    localworker1_celery_app,
    globalworker_celery_app,
):
    request_id = "test_worker_info_" + uuid.uuid4().hex + "_request"

    task_signature = get_celery_task_signature("get_worker_info")
    localworker1_async_result = localworker1_celery_app.queue_task(
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
    )
    localworker1_task_response = localworker1_celery_app.get_result(
        async_result=localworker1_async_result,
        logger=StdOutputLogger(),
        timeout=TASKS_TIMEOUT,
    )

    worker_info = WorkerInfo.parse_raw(localworker1_task_response)

    assert worker_info.id == worker_info_per_worker["testlocalworker1"].id
    assert worker_info.role == worker_info_per_worker["testlocalworker1"].role

    globalworker_async_result = globalworker_celery_app.queue_task(
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
    )
    globalworker_task_response = globalworker_celery_app.get_result(
        async_result=globalworker_async_result,
        logger=StdOutputLogger(),
        timeout=TASKS_TIMEOUT,
    )

    worker_info = WorkerInfo.parse_raw(globalworker_task_response)

    assert worker_info.id == worker_info_per_worker["testglobalworker"].id
    assert worker_info.role == worker_info_per_worker["testglobalworker"].role

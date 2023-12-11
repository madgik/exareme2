import uuid as uuid

import pytest

from tests.standalone_tests.conftest import TASKS_TIMEOUT
from tests.standalone_tests.nodes_communication_helper import get_celery_task_signature
from tests.standalone_tests.std_output_logger import StdOutputLogger

healthcheck_task_signature = get_celery_task_signature("healthcheck")


@pytest.fixture(autouse=True)
def request_id():
    return "test_healthcheck_" + uuid.uuid4().hex


@pytest.mark.slow
def test_healthcheck_task(
    request_id,
    localnode1_node_service,
    localnode1_celery_app,
):
    logger = StdOutputLogger()
    async_result = localnode1_celery_app.queue_task(
        task_signature=healthcheck_task_signature,
        logger=logger,
        request_id=request_id,
        check_db=True,
    )
    try:
        localnode1_celery_app.get_result(
            async_result=async_result,
            timeout=TASKS_TIMEOUT,
            logger=logger,
        )
    except Exception as exc:
        pytest.fail(f"Healthcheck failed with error: {exc}")

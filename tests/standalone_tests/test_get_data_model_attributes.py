import uuid

import pytest

from exareme2.node_tasks_DTOs import DataModelAttributes
from tests.standalone_tests.conftest import TASKS_TIMEOUT
from tests.standalone_tests.nodes_communication_helper import get_celery_task_signature
from tests.standalone_tests.std_output_logger import StdOutputLogger


def get_data_model():
    return [
        "dementia:0.1",
        "tbi:0.1",
    ]


@pytest.mark.slow
@pytest.mark.parametrize(
    "data_model",
    get_data_model(),
)
def test_get_data_model_attributes(
    data_model,
    localnode1_node_service,
    localnode1_celery_app,
    load_data_localnode1,
):
    request_id = "test_attributes_" + uuid.uuid4().hex + "_request"

    task_signature = get_celery_task_signature("get_data_model_attributes")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        data_model=data_model,
    )
    data_model_attributes_json = localnode1_celery_app.get_result(
        async_result=async_result,
        logger=StdOutputLogger(),
        timeout=TASKS_TIMEOUT,
    )
    attributes = DataModelAttributes.parse_raw(data_model_attributes_json)
    assert (
        f"{attributes.properties['cdes']['code']}:{attributes.properties['cdes']['version']}"
        == data_model
    )

import uuid

import pytest

from exareme2.worker_communication import DataModelMetadata
from exareme2.worker_communication import parse_data_model_metadata
from tests.standalone_tests.conftest import TASKS_TIMEOUT
from tests.standalone_tests.controller.workers_communication_helper import (
    get_celery_task_signature,
)
from tests.standalone_tests.std_output_logger import StdOutputLogger


@pytest.mark.slow
def test_get_data_model_metadata_and_dataset_locations(
    localworker1_worker_service,
    localworker1_celery_app,
    load_data_localworker1,
):
    request_id = "test_metadata_" + uuid.uuid4().hex + "_request"

    task_signature = get_celery_task_signature(
        "get_worker_data_model_metadata_and_datasets"
    )
    async_result = localworker1_celery_app.queue_task(
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
    )
    data_models_metadata, datasets_per_data_model = localworker1_celery_app.get_result(
        async_result=async_result,
        logger=StdOutputLogger(),
        timeout=TASKS_TIMEOUT,
    )
    assert data_models_metadata is not None
    assert datasets_per_data_model is not None
    assert "dementia:0.1" in data_models_metadata
    assert "dementia:0.1" in datasets_per_data_model
    assert datasets_per_data_model.keys() == data_models_metadata.keys()
    assert datasets_per_data_model["dementia:0.1"]
    assert data_models_metadata["dementia:0.1"]
    assert datasets_per_data_model["dementia:0.1"] == [
        "desd-synthdata0",
        "desd-synthdata1",
        "desd-synthdata2",
        "desd-synthdata3",
        "edsd0",
        "edsd1",
        "edsd2",
        "edsd3",
        "ppmi0",
        "ppmi1",
        "ppmi2",
        "ppmi3",
    ]
    assert isinstance(
        parse_data_model_metadata(data_models_metadata["dementia:0.1"]),
        DataModelMetadata,
    )

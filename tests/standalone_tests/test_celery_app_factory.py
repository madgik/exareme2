import random
from unittest.mock import patch

import celery
import pytest

from exareme2 import AttrDict
from exareme2 import DType
from exareme2.controller.celery.app import CeleryConnectionError
from exareme2.controller.celery.app import CeleryWrapper
from exareme2.worker_communication import ColumnInfo
from exareme2.worker_communication import TableSchema
from tests.standalone_tests.conftest import COMMON_IP
from tests.standalone_tests.conftest import RABBITMQ_LOCALWORKER1_PORT
from tests.standalone_tests.conftest import RABBITMQ_LOCALWORKERTMP_PORT
from tests.standalone_tests.conftest import kill_service
from tests.standalone_tests.conftest import remove_localworkertmp_rabbitmq
from tests.standalone_tests.std_output_logger import StdOutputLogger

REQUEST_ID = "testrequestid"


@pytest.fixture(scope="session")
def controller_config_dict_mock():
    controller_config = {
        "log_level": "DEBUG",
        "framework_log_level": "INFO",
        "rabbitmq": {
            "user": "user",
            "password": "password",
            "vhost": "user_vhost",
        },
    }

    return controller_config


@pytest.fixture(autouse=True, scope="session")
def patch_celery_app(controller_config_dict_mock):
    with patch(
        "exareme2.controller.celery.app.controller_config",
        AttrDict(controller_config_dict_mock),
    ):
        yield


@pytest.fixture(scope="session")
def task_signatures():
    return {
        "create_table": "exareme2.worker.exareme2.tables.tables_api.create_table",
    }


# TODO: create slow udf to provoke TimeoutException


@pytest.mark.slow
def test_queue_task(localworker1_worker_service, task_signatures):
    socket_addr = f"{COMMON_IP}:{RABBITMQ_LOCALWORKER1_PORT}"
    celery_app = CeleryWrapper(socket_addr=socket_addr)

    test_table_schema = TableSchema(
        columns=[
            ColumnInfo(name="var1", dtype=DType.INT),
            ColumnInfo(name="var2", dtype=DType.STR),
        ]
    )
    async_result = celery_app.queue_task(
        task_signature=task_signatures["create_table"],
        logger=StdOutputLogger(),
        request_id=REQUEST_ID,
        context_id=get_a_random_context_id(),
        command_id="testcmndid",
        schema_json=test_table_schema.json(),
    )
    assert isinstance(async_result, celery.result.AsyncResult)


@pytest.mark.slow
def test_get_result(localworker1_worker_service, task_signatures):
    socket_addr = f"{COMMON_IP}:{RABBITMQ_LOCALWORKER1_PORT}"
    celery_app = CeleryWrapper(socket_addr=socket_addr)

    test_table_schema = TableSchema(
        columns=[
            ColumnInfo(name="var1", dtype=DType.INT),
            ColumnInfo(name="var2", dtype=DType.STR),
        ]
    )
    async_result = celery_app.queue_task(
        task_signature=task_signatures["create_table"],
        logger=StdOutputLogger(),
        request_id=REQUEST_ID,
        context_id=get_a_random_context_id(),
        command_id="testcmndid",
        schema_json=test_table_schema.json(),
    )

    result = celery_app.get_result(
        async_result=async_result,
        logger=StdOutputLogger(),
        timeout=30,
    )

    assert isinstance(result, str)


@pytest.mark.slow
@pytest.mark.very_slow
def test_queue_task_node_down(localworkertmp_worker_service, task_signatures):
    socket_addr = f"{COMMON_IP}:{RABBITMQ_LOCALWORKERTMP_PORT}"
    celery_app = CeleryWrapper(socket_addr=socket_addr)

    remove_localworkertmp_rabbitmq()

    test_table_schema = TableSchema(
        columns=[
            ColumnInfo(name="var1", dtype=DType.INT),
            ColumnInfo(name="var2", dtype=DType.STR),
        ]
    )

    with pytest.raises(CeleryConnectionError):
        celery_app.queue_task(
            task_signature=task_signatures["create_table"],
            logger=StdOutputLogger(),
            request_id=REQUEST_ID,
            context_id=get_a_random_context_id(),
            command_id="testcmndid",
            schema_json=test_table_schema.json(),
        )


@pytest.mark.slow
@pytest.mark.very_slow
def test_get_result_node_down(localworkertmp_worker_service, task_signatures):
    socket_addr = f"{COMMON_IP}:{RABBITMQ_LOCALWORKERTMP_PORT}"
    celery_app = CeleryWrapper(socket_addr=socket_addr)

    test_table_schema = TableSchema(
        columns=[
            ColumnInfo(name="var1", dtype=DType.INT),
            ColumnInfo(name="var2", dtype=DType.STR),
        ]
    )

    async_result = celery_app.queue_task(
        task_signature=task_signatures["create_table"],
        logger=StdOutputLogger(),
        request_id=REQUEST_ID,
        context_id=get_a_random_context_id(),
        command_id="testcmndid",
        schema_json=test_table_schema.json(),
    )

    kill_service(localworkertmp_worker_service)
    remove_localworkertmp_rabbitmq()

    with pytest.raises(CeleryConnectionError):
        result = celery_app.get_result(
            async_result=async_result, logger=StdOutputLogger(), timeout=5
        )


def get_a_random_context_id() -> str:
    return str(random.randint(1, 99999))

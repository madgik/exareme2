import uuid as uuid
from unittest.mock import patch

import pytest

from exareme2 import AttrDict
from exareme2.controller import logger as ctrl_logger
from exareme2.controller.celery.app import CeleryConnectionError
from exareme2.controller.services import WorkerLandscapeAggregator
from exareme2.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    _wlaRegistries,
)
from exareme2.controller.workers_addresses import WorkersAddresses
from exareme2.controller.workers_addresses import WorkersAddressesFactory
from tests.standalone_tests.conftest import RABBITMQ_LOCALWORKER1_ADDR
from tests.standalone_tests.conftest import RABBITMQ_LOCALWORKERTMP_ADDR
from tests.standalone_tests.conftest import TASKS_TIMEOUT
from tests.standalone_tests.controller.services.worker_landscape_agreggator.test_worker_landscape_aggregator_update_loop import (
    CustomWorkerAddresses,
)
from tests.standalone_tests.controller.workers_communication_helper import (
    get_celery_task_signature,
)
from tests.standalone_tests.std_output_logger import StdOutputLogger

healthcheck_task_signature = get_celery_task_signature("healthcheck")


@pytest.fixture(autouse=True)
def request_id():
    return "test_healthcheck_" + uuid.uuid4().hex


@pytest.mark.slow
def test_healthcheck_task(
    request_id,
    localworker1_worker_service,
    localworker1_celery_app,
):
    logger = StdOutputLogger()
    async_result = localworker1_celery_app.queue_task(
        task_signature=healthcheck_task_signature,
        logger=logger,
        request_id=request_id,
        check_db=True,
    )
    try:
        localworker1_celery_app.get_result(
            async_result=async_result,
            timeout=TASKS_TIMEOUT,
            logger=logger,
        )
    except Exception as exc:
        pytest.fail(f"Healthcheck failed with error: {exc}")


@pytest.fixture(scope="session")
def controller_config():
    controller_config = {
        "rabbitmq": {
            "user": "user",
            "password": "password",
            "vhost": "user_vhost",
            "celery_tasks_timeout": 10,
            "celery_run_udf_task_timeout": 30,
            "celery_tasks_max_retries": 3,
            "celery_tasks_interval_start": 0,
            "celery_tasks_interval_step": 0.2,
            "celery_tasks_interval_max": 0.5,
        },
    }
    return controller_config


def get_custom_workers_addresses_localworker1() -> WorkersAddresses:
    return CustomWorkerAddresses([RABBITMQ_LOCALWORKER1_ADDR])


def get_custom_workers_addresses_localworkertmp() -> WorkersAddresses:
    return CustomWorkerAddresses([RABBITMQ_LOCALWORKERTMP_ADDR])


@pytest.fixture(autouse=True, scope="function")
def patch_workers_addresses():
    with patch.object(
        WorkersAddressesFactory,
        "get_workers_addresses",
    ) as patched:
        yield patched


@pytest.fixture(scope="function")
def worker_landscape_aggregator(controller_config):
    controller_config = AttrDict(controller_config)

    worker_landscape_aggregator = WorkerLandscapeAggregator(
        logger=ctrl_logger.get_background_service_logger(),
        update_interval=0,
        tasks_timeout=controller_config.rabbitmq.celery_tasks_timeout,
        run_udf_task_timeout=controller_config.rabbitmq.celery_run_udf_task_timeout,
        deployment_type=None,
        localworkers=None,
    )
    worker_landscape_aggregator.stop()
    worker_landscape_aggregator.keep_updating = False
    worker_landscape_aggregator._wla_registries = _wlaRegistries()

    return worker_landscape_aggregator


@pytest.mark.slow
def test_healthcheck_success(
    patch_workers_addresses,
    localworker1_worker_service,
    worker_landscape_aggregator,
):
    patch_workers_addresses.side_effect = get_custom_workers_addresses_localworker1

    try:
        worker_landscape_aggregator.healthcheck()
    except Exception as exc:
        pytest.fail(f"Healthcheck failed with exception: {exc}")


def test_healthcheck_fail(
    patch_workers_addresses,
    worker_landscape_aggregator,
):
    patch_workers_addresses.side_effect = get_custom_workers_addresses_localworkertmp

    with pytest.raises(CeleryConnectionError):
        worker_landscape_aggregator.healthcheck()

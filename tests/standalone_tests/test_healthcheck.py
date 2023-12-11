import uuid as uuid
from unittest.mock import patch

import pytest

from exareme2 import AttrDict
from exareme2.controller import logger as ctrl_logger
from exareme2.controller.celery.app import CeleryConnectionError
from exareme2.controller.nodes_addresses import NodesAddresses
from exareme2.controller.nodes_addresses import NodesAddressesFactory
from exareme2.controller.services import NodeLandscapeAggregator
from exareme2.controller.services.node_landscape_aggregator import _NLARegistries
from tests.standalone_tests.conftest import RABBITMQ_LOCALNODE1_ADDR
from tests.standalone_tests.conftest import RABBITMQ_LOCALNODETMP_ADDR
from tests.standalone_tests.conftest import TASKS_TIMEOUT
from tests.standalone_tests.nodes_communication_helper import get_celery_task_signature
from tests.standalone_tests.std_output_logger import StdOutputLogger
from tests.standalone_tests.test_node_landscape_aggregator_update_loop import (
    CustomNodeAddresses,
)

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


def get_custom_nodes_addresses_localnode1() -> NodesAddresses:
    return CustomNodeAddresses([RABBITMQ_LOCALNODE1_ADDR])


def get_custom_nodes_addresses_localnodetmp() -> NodesAddresses:
    return CustomNodeAddresses([RABBITMQ_LOCALNODETMP_ADDR])


@pytest.fixture(autouse=True, scope="function")
def patch_nodes_addresses():
    with patch.object(
        NodesAddressesFactory,
        "get_nodes_addresses",
    ) as patched:
        yield patched


@pytest.fixture(scope="function")
def node_landscape_aggregator(controller_config):
    controller_config = AttrDict(controller_config)

    node_landscape_aggregator = NodeLandscapeAggregator(
        logger=ctrl_logger.get_background_service_logger(),
        update_interval=0,
        tasks_timeout=controller_config.rabbitmq.celery_tasks_timeout,
        run_udf_task_timeout=controller_config.rabbitmq.celery_run_udf_task_timeout,
        deployment_type=None,
        localnodes=None,
    )
    node_landscape_aggregator.stop()
    node_landscape_aggregator.keep_updating = False
    node_landscape_aggregator._nla_registries = _NLARegistries()

    return node_landscape_aggregator


@pytest.mark.slow
def test_healthcheck_success(
    patch_nodes_addresses,
    localnode1_node_service,
    node_landscape_aggregator,
):
    patch_nodes_addresses.side_effect = get_custom_nodes_addresses_localnode1

    try:
        node_landscape_aggregator.healthcheck()
    except Exception as exc:
        pytest.fail(f"Healthcheck failed with exception: {exc}")


def test_healthcheck_fail(
    patch_nodes_addresses,
    node_landscape_aggregator,
):
    patch_nodes_addresses.side_effect = get_custom_nodes_addresses_localnodetmp

    with pytest.raises(CeleryConnectionError):
        node_landscape_aggregator.healthcheck()

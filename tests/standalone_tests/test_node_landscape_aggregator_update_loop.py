import time
from os import path
from unittest.mock import patch

import pytest
import toml

from mipengine import AttrDict
from mipengine.controller import controller_logger as ctrl_logger
from mipengine.controller.node_landscape_aggregator import NodeLandscapeAggregator
from tests.standalone_tests.conftest import LOCALNODE1_CONFIG_FILE
from tests.standalone_tests.conftest import LOCALNODE2_CONFIG_FILE
from tests.standalone_tests.conftest import TEST_ENV_CONFIG_FOLDER

WAIT_TIME_LIMIT = 120
WAIT_TIME_NLA_UPDATE = 5


@pytest.fixture(scope="session")
def controller_config_mock():
    controller_config = AttrDict(
        {
            "log_level": "DEBUG",
            "framework_log_level": "INFO",
            "deployment_type": "LOCAL",
            "node_landscape_aggregator_update_interval": WAIT_TIME_NLA_UPDATE,
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
            "smpc": {
                "enabled": False,
                "optional": False,
                "coordinator_address": "$SMPC_COORDINATOR_ADDRESS",
            },
        }
    )
    return controller_config


@pytest.fixture(autouse=True, scope="session")
def patch_node_landscape_aggregator(controller_config_mock):
    with patch(
        "mipengine.controller.node_landscape_aggregator.NODE_LANDSCAPE_AGGREGATOR_UPDATE_INTERVAL",
        controller_config_mock.node_landscape_aggregator_update_interval,
    ), patch(
        "mipengine.controller.node_landscape_aggregator.CELERY_TASKS_TIMEOUT",
        controller_config_mock.rabbitmq.celery_tasks_timeout,
    ):
        yield


@pytest.fixture(autouse=True, scope="session")
def init_background_controller_logger():
    ctrl_logger.set_background_service_logger("DEBUG")


def _get_nodes_addresses_empty():
    return []


def _get_nodes_addresses_one_local():
    return ["172.17.0.1:60000", "172.17.0.1:60001"]


def _get_nodes_addresses_two_locals():
    return ["172.17.0.1:60000", "172.17.0.1:60001", "172.17.0.1:60002"]


@pytest.fixture(autouse=True, scope="session")
def patch_celery_app(controller_config_mock):
    with patch(
        "mipengine.controller.celery_app.controller_config", controller_config_mock
    ):
        yield


@pytest.fixture(autouse=True, scope="function")
def patch_get_nodes_addresses():
    with patch(
        "mipengine.controller.node_landscape_aggregator.get_nodes_addresses",
        side_effect=_get_nodes_addresses_one_local,
    ) as patch_get_nodes_addresses:
        yield patch_get_nodes_addresses


@pytest.mark.slow
def test_update_loop_data_properly_added(
    patch_get_nodes_addresses,
    globalnode_node_service,
    localnode1_node_service,
    load_data_localnode1,
    localnode2_node_service,
    load_data_localnode2,
    reset_node_landscape_aggregator,
):
    node_landscape_aggregator = NodeLandscapeAggregator()
    node_landscape_aggregator.start()

    # wait until NLA's NodeRegistry to contain the localnode1
    start = time.time()
    while not (
        node_landscape_aggregator.get_cdes_per_data_model().values
        and "tbi:0.1" in node_landscape_aggregator.get_cdes_per_data_model().values
        and "dementia:0.1" in node_landscape_aggregator.get_cdes_per_data_model().values
    ):
        if time.time() - start > WAIT_TIME_LIMIT:
            pytest.fail(
                f"NLA did not update DataModelRegistry properly during {WAIT_TIME_LIMIT=}"
            )
        time.sleep(1)

    node_landscape_aggregator.stop()


@pytest.mark.slow
def test_update_loop_get_node_info_fail(
    patch_get_nodes_addresses,
    globalnode_node_service,
    localnode1_node_service,
    load_data_localnode1,
    localnode2_node_service,
    load_data_localnode2,
    reset_node_landscape_aggregator,
):
    patch_get_nodes_addresses.side_effect = _get_nodes_addresses_empty
    node_landscape_aggregator = NodeLandscapeAggregator()
    node_landscape_aggregator.start()

    # NLA will never contain nodes
    start = time.time()
    while time.time() - start < WAIT_TIME_NLA_UPDATE:
        if node_landscape_aggregator.get_nodes():
            pytest.fail(
                f"Node registry should not contain {node_landscape_aggregator.get_nodes()}"
            )
        time.sleep(1)
    node_landscape_aggregator.stop()


@pytest.mark.slow
def test_update_loop_nodes_properly_added(
    patch_get_nodes_addresses,
    globalnode_node_service,
    localnode1_node_service,
    load_data_localnode1,
    localnode2_node_service,
    load_data_localnode2,
    reset_node_landscape_aggregator,
):
    localnode1_node_id = get_localnode1_node_id()
    localnode2_node_id = get_localnode2_node_id()
    node_landscape_aggregator = NodeLandscapeAggregator()
    node_landscape_aggregator.start()

    # wait until NLA's NodeRegistry to contain the localnode1
    start = time.time()
    while not any(
        [
            node.id == localnode1_node_id
            for node in node_landscape_aggregator.get_all_local_nodes()
        ]
    ):
        if time.time() - start > WAIT_TIME_LIMIT:
            pytest.fail(
                f"Node registry did not contain the localnode1 during {WAIT_TIME_LIMIT=}"
            )

        time.sleep(1)

    patch_get_nodes_addresses.side_effect = _get_nodes_addresses_two_locals

    # wait until NLA's NodeRegistry to contain the localnode2
    start = time.time()
    while not any(
        [
            node.id == localnode2_node_id
            for node in node_landscape_aggregator.get_nodes()
        ]
    ):
        if time.time() - start > WAIT_TIME_LIMIT:
            pytest.fail(
                f"Node registry did not contain the localnode2 during {WAIT_TIME_LIMIT=}"
            )
        time.sleep(1)

    node_landscape_aggregator.stop()


def get_localnode2_node_id():
    local_node_filepath = path.join(TEST_ENV_CONFIG_FOLDER, LOCALNODE2_CONFIG_FILE)
    with open(local_node_filepath) as fp:
        tmp = toml.load(fp)
        node_id = tmp["identifier"]
    return node_id


def get_localnode1_node_id():
    local_node_filepath = path.join(TEST_ENV_CONFIG_FOLDER, LOCALNODE1_CONFIG_FILE)
    with open(local_node_filepath) as fp:
        tmp = toml.load(fp)
        node_id = tmp["identifier"]
    return node_id

from os import path
from unittest.mock import patch

import pytest
import toml

from mipengine import AttrDict
from mipengine.controller import controller_logger as ctrl_logger
from mipengine.controller.node_landscape_aggregator import NodeLandscapeAggregator
from mipengine.controller.nodes_addresses import NodesAddresses
from mipengine.controller.nodes_addresses import NodesAddressesFactory
from tests.standalone_tests.conftest import LOCALNODE1_CONFIG_FILE
from tests.standalone_tests.conftest import LOCALNODE2_CONFIG_FILE
from tests.standalone_tests.conftest import TEST_ENV_CONFIG_FOLDER

WAIT_TIME_LIMIT = 120
WAIT_TIME_NLA_UPDATE = 5
CELERY_TASKS_TIMEOUT = 10


@pytest.fixture(autouse=True, scope="session")
def patch_node_landscape_aggregator():
    with patch(
        "mipengine.controller.node_landscape_aggregator.NODE_LANDSCAPE_AGGREGATOR_UPDATE_INTERVAL",
        WAIT_TIME_NLA_UPDATE,
    ), patch(
        "mipengine.controller.node_landscape_aggregator.CELERY_TASKS_TIMEOUT",
        CELERY_TASKS_TIMEOUT,
    ):
        yield


@pytest.fixture(autouse=True, scope="session")
def init_background_controller_logger():
    ctrl_logger.set_background_service_logger("DEBUG")


@pytest.fixture(autouse=True, scope="session")
def patch_celery_app():
    with patch(
        "mipengine.controller.celery_app.controller_config",
        AttrDict(
            {
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
                }
            }
        ),
    ):
        yield


class CustomNodeAddresses(NodesAddresses):
    def __init__(self, socket_addresses):
        self._socket_addresses = socket_addresses


def get_custom_nodes_addresses_without_nodes() -> NodesAddresses:
    return CustomNodeAddresses([])


def get_custom_nodes_addresses() -> NodesAddresses:
    return CustomNodeAddresses(["172.17.0.1:60000", "172.17.0.1:60001"])


def get_custom_nodes_addresses_1_2() -> NodesAddresses:
    return CustomNodeAddresses(
        ["172.17.0.1:60000", "172.17.0.1:60001", "172.17.0.1:60002"]
    )


@pytest.fixture(autouse=True, scope="function")
def patch_nodes_addresses():
    with patch.object(
        NodesAddressesFactory,
        "get_nodes_addresses",
        side_effect=get_custom_nodes_addresses,
    ) as patched:
        yield patched


@pytest.mark.slow
def test_update_loop_data_properly_added(
    patch_nodes_addresses,
    globalnode_node_service,
    localnode1_node_service,
    load_data_localnode1,
    reset_node_landscape_aggregator,
):
    node_landscape_aggregator = NodeLandscapeAggregator()
    node_landscape_aggregator._update()

    assert (
        node_landscape_aggregator.get_cdes_per_data_model().data_models_cdes
        and "tbi:0.1"
        in node_landscape_aggregator.get_cdes_per_data_model().data_models_cdes
        and "dementia:0.1"
        in node_landscape_aggregator.get_cdes_per_data_model().data_models_cdes
    )


@pytest.mark.slow
def test_update_loop_get_node_info_fail(
    patch_nodes_addresses,
    reset_node_landscape_aggregator,
):

    node_landscape_aggregator = NodeLandscapeAggregator()
    node_landscape_aggregator._update()
    assert node_landscape_aggregator.get_nodes()

    patch_nodes_addresses.side_effect = get_custom_nodes_addresses_without_nodes
    node_landscape_aggregator._update()

    assert not node_landscape_aggregator.get_nodes()


@pytest.mark.slow
def test_update_loop_nodes_properly_added(
    patch_nodes_addresses,
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
    node_landscape_aggregator._update()
    assert any(
        [
            node.id == localnode1_node_id
            for node in node_landscape_aggregator.get_all_local_nodes()
        ]
    )

    assert all(
        [
            node.id != localnode2_node_id
            for node in node_landscape_aggregator.get_all_local_nodes()
        ]
    )

    patch_nodes_addresses.side_effect = get_custom_nodes_addresses_1_2
    node_landscape_aggregator._update()

    assert any(
        [
            node.id == localnode2_node_id
            for node in node_landscape_aggregator.get_all_local_nodes()
        ]
    )


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

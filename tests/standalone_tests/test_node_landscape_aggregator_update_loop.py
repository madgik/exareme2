from os import path
from unittest.mock import patch

import pytest
import toml

from exareme2 import AttrDict
from exareme2.controller import logger as ctrl_logger
from exareme2.controller.nodes_addresses import NodesAddresses
from exareme2.controller.nodes_addresses import NodesAddressesFactory
from exareme2.controller.services.node_landscape_aggregator import (
    NodeLandscapeAggregator,
)
from exareme2.controller.services.node_landscape_aggregator import _NLARegistries
from tests.standalone_tests.conftest import GLOBALNODE_CONFIG_FILE
from tests.standalone_tests.conftest import LOCALNODE1_CONFIG_FILE
from tests.standalone_tests.conftest import LOCALNODE2_CONFIG_FILE
from tests.standalone_tests.conftest import LOCALNODETMP_CONFIG_FILE
from tests.standalone_tests.conftest import TEST_ENV_CONFIG_FOLDER


@pytest.fixture(scope="session")
def controller_config():
    controller_config = {
        "deployment_type": "LOCAL",
        "node_landscape_aggregator_update_interval": 30,
        "localnodes": {
            "config_file": "./tests/standalone_tests/testing_env_configs/test_globalnode_localnode1_localnode2_localnodetmp_addresses.json"
        },
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
        "smpc": {"enabled": False, "optional": False},
    }
    return controller_config


@pytest.fixture(autouse=True, scope="session")
def init_background_controller_logger():
    ctrl_logger.set_background_service_logger("DEBUG")


@pytest.fixture(autouse=True, scope="session")
def patch_celery_app(controller_config):
    with patch(
        "exareme2.controller.celery.app.controller_config", AttrDict(controller_config)
    ):
        yield


class CustomNodeAddresses(NodesAddresses):
    def __init__(self, socket_addresses):
        self._socket_addresses = socket_addresses


def get_custom_nodes_addresses_without_nodes() -> NodesAddresses:
    return CustomNodeAddresses([])


def get_custom_nodes_addresses_global_and_tmp() -> NodesAddresses:
    return CustomNodeAddresses(["172.17.0.1:60000", "172.17.0.1:60003"])


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


@pytest.fixture(scope="function")
def node_landscape_aggregator(controller_config):
    controller_config = AttrDict(controller_config)

    node_landscape_aggregator = NodeLandscapeAggregator(
        logger=ctrl_logger.get_background_service_logger(),
        update_interval=controller_config.node_landscape_aggregator_update_interval,
        tasks_timeout=controller_config.rabbitmq.celery_tasks_timeout,
        run_udf_task_timeout=controller_config.rabbitmq.celery_run_udf_task_timeout,
        deployment_type=controller_config.deployment_type,
        localnodes=controller_config.localnodes,
    )
    node_landscape_aggregator.stop()
    node_landscape_aggregator.keep_updating = False
    node_landscape_aggregator._nla_registries = _NLARegistries()

    return node_landscape_aggregator


@pytest.mark.slow
def test_update_loop_data_properly_added(
    patch_nodes_addresses,
    globalnode_node_service,
    localnode1_node_service,
    load_data_localnode1,
    node_landscape_aggregator,
):
    node_landscape_aggregator.update()

    assert (
        node_landscape_aggregator.get_cdes_per_data_model().data_models_cdes
        and "tbi:0.1"
        in node_landscape_aggregator.get_cdes_per_data_model().data_models_cdes
        and "dementia:0.1"
        in node_landscape_aggregator.get_cdes_per_data_model().data_models_cdes
    )


@pytest.mark.slow
def test_update_loop_get_node_info_fail(
    patch_nodes_addresses, globalnode_node_service, node_landscape_aggregator
):
    patch_nodes_addresses.side_effect = get_custom_nodes_addresses_global_and_tmp
    node_landscape_aggregator.update()
    assert node_landscape_aggregator.get_nodes()
    assert any(
        [
            node.id == get_globalnode_node_id()
            for node in node_landscape_aggregator.get_nodes()
        ]
    )

    assert all(
        [
            node.id != get_localnodetmp_node_id()
            for node in node_landscape_aggregator.get_all_local_nodes()
        ]
    )


@pytest.mark.slow
def test_update_loop_nodes_properly_added(
    patch_nodes_addresses,
    globalnode_node_service,
    localnode1_node_service,
    load_data_localnode1,
    localnode2_node_service,
    load_data_localnode2,
    node_landscape_aggregator,
):
    localnode1_node_id = get_localnode1_node_id()
    localnode2_node_id = get_localnode2_node_id()
    node_landscape_aggregator.update()
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
    node_landscape_aggregator.update()

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


def get_localnodetmp_node_id():
    local_node_filepath = path.join(TEST_ENV_CONFIG_FOLDER, LOCALNODETMP_CONFIG_FILE)
    with open(local_node_filepath) as fp:
        tmp = toml.load(fp)
        node_id = tmp["identifier"]
    return node_id


def get_globalnode_node_id():
    local_node_filepath = path.join(TEST_ENV_CONFIG_FOLDER, GLOBALNODE_CONFIG_FILE)
    with open(local_node_filepath) as fp:
        tmp = toml.load(fp)
        node_id = tmp["identifier"]
    return node_id

import asyncio
import time
from os import path
from unittest.mock import patch

import pytest
import toml

from mipengine import AttrDict
from mipengine.controller import controller_logger as ctrl_logger
from mipengine.controller.api.algorithm_request_dto import AlgorithmInputDataDTO
from mipengine.controller.api.algorithm_request_dto import AlgorithmRequestDTO
from mipengine.controller.controller import Controller
from mipengine.controller.controller import get_a_uniqueid
from tests.standalone_tests.conftest import ALGORITHM_FOLDERS_ENV_VARIABLE_VALUE
from tests.standalone_tests.conftest import LOCALNODETMP_CONFIG_FILE
from tests.standalone_tests.conftest import RABBITMQ_LOCALNODETMP_NAME
from tests.standalone_tests.conftest import RABBITMQ_LOCALNODETMP_PORT
from tests.standalone_tests.conftest import TEST_ENV_CONFIG_FOLDER
from tests.standalone_tests.conftest import _create_node_service
from tests.standalone_tests.conftest import _create_rabbitmq_container
from tests.standalone_tests.conftest import kill_node_service
from tests.standalone_tests.conftest import remove_localnodetmp_rabbitmq

MAX_RETRIES = 30


@pytest.fixture(scope="session")
def controller_config_mock():
    controller_config = AttrDict(
        {
            "log_level": "DEBUG",
            "framework_log_level": "INFO",
            "deployment_type": "LOCAL",
            "node_landscape_aggregator_update_interval": 2,  # 5,
            "nodes_cleanup_interval": 2,
            "localnodes": {
                "config_file": "./tests/standalone_tests/testing_env_configs/test_localnodes_addresses.json",
                "dns": "",
                "port": "",
            },
            "rabbitmq": {
                "user": "user",
                "password": "password",
                "vhost": "user_vhost",
                "celery_tasks_timeout": 30,  # 60,
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
def patch_controller(controller_config_mock):
    with patch(
        "mipengine.controller.controller.controller_config",
        controller_config_mock,
    ):
        yield


@pytest.fixture(autouse=True, scope="session")
def patch_node_landscape_aggregator(controller_config_mock):
    with patch(
        "mipengine.controller.node_landscape_aggregator.controller_config",
        controller_config_mock,
    ), patch(
        "mipengine.controller.node_landscape_aggregator.NODE_LANDSCAPE_AGGREGATOR_UPDATE_INTERVAL",
        controller_config_mock.node_landscape_aggregator_update_interval,
    ), patch(
        "mipengine.controller.node_landscape_aggregator.CELERY_TASKS_TIMEOUT",
        controller_config_mock.rabbitmq.celery_tasks_timeout,
    ):
        yield


@pytest.fixture(autouse=True, scope="session")
def patch_celery_app(controller_config_mock):
    with patch(
        "mipengine.controller.celery_app.controller_config", controller_config_mock
    ):
        yield


@pytest.fixture(autouse=True, scope="session")
def patch_algorithm_executor(controller_config_mock):
    with patch(
        "mipengine.controller.algorithm_executor.ctrl_config", controller_config_mock
    ):
        yield


@pytest.fixture(autouse=True, scope="session")
def patch_node_address(controller_config_mock):
    with patch(
        "mipengine.controller.node_address.controller_config", controller_config_mock
    ):
        yield


@pytest.mark.slow
@pytest.mark.asyncio
async def test_node_registry_node_service_down(
    load_data_localnodetmp,
    globalnode_node_service,
    localnodetmp_node_service,
):

    # get tmp localnode node_id from config file
    localnodetmp_node_id = get_localnodetmp_node_id()
    controller = Controller()

    # wait until node registry gets the nodes info
    await controller.start_node_landscape_aggregator()

    for _ in range(MAX_RETRIES):
        if localnodetmp_node_id in controller.get_all_local_nodes():
            break
        await asyncio.sleep(2)
    else:
        pytest.fail(
            "Exceeded max retries while waiting for the node registry to contain the tmplocalnode"
        )

    kill_node_service(localnodetmp_node_service)

    # wait until node registry removes tmplocalnode
    for _ in range(MAX_RETRIES):
        if localnodetmp_node_id not in controller.get_all_local_nodes():
            break
        await asyncio.sleep(2)
    else:
        pytest.fail(
            "Exceeded max retries while waiting for the node registry to remove the tmplocalnode"
        )

    # restart tmplocalnode node service (the celery app)
    localnodetmp_node_service_proc = start_localnodetmp_node_service()

    # wait until node registry re-added tmplocalnode
    for _ in range(MAX_RETRIES):
        if localnodetmp_node_id in controller.get_all_local_nodes():
            break
        await asyncio.sleep(2)
    else:
        pytest.fail(
            "Exceeded max retries while waiting for the node registry to re-add the tmplocalnode"
        )

    await controller.stop_node_landscape_aggregator()

    # the node service was started in here, so it must manually be killed, otherwise it is
    # alive through the whole pytest session and is erroneously accessed by other tests
    # where the node service is supposedly down
    kill_node_service(localnodetmp_node_service_proc)


def get_localnodetmp_node_id():
    local_node_filepath = path.join(TEST_ENV_CONFIG_FOLDER, LOCALNODETMP_CONFIG_FILE)
    with open(local_node_filepath) as fp:
        tmp = toml.load(fp)
        node_id = tmp["identifier"]
    return node_id


def start_localnodetmp_node_service():
    node_config_file = LOCALNODETMP_CONFIG_FILE
    algo_folders_env_variable_val = ALGORITHM_FOLDERS_ENV_VARIABLE_VALUE
    node_config_filepath = path.join(TEST_ENV_CONFIG_FOLDER, node_config_file)
    proc = _create_node_service(algo_folders_env_variable_val, node_config_filepath)
    return proc

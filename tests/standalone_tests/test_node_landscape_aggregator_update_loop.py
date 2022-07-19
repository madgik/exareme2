import time
from os import path
from unittest.mock import patch

import pytest
import toml

from mipengine import AttrDict
from mipengine.controller import controller_logger as ctrl_logger
from mipengine.controller.node_landscape_aggregator import NodeLandscapeAggregator
from tests.standalone_tests.conftest import ALGORITHM_FOLDERS_ENV_VARIABLE_VALUE
from tests.standalone_tests.conftest import LOCALNODETMP_CONFIG_FILE
from tests.standalone_tests.conftest import RABBITMQ_LOCALNODETMP_NAME
from tests.standalone_tests.conftest import RABBITMQ_LOCALNODETMP_PORT
from tests.standalone_tests.conftest import TEST_ENV_CONFIG_FOLDER
from tests.standalone_tests.conftest import _create_node_service
from tests.standalone_tests.conftest import _create_rabbitmq_container
from tests.standalone_tests.conftest import _load_data_monetdb_container
from tests.standalone_tests.conftest import _remove_data_model_from_localnodetmp_monetdb
from tests.standalone_tests.conftest import kill_service
from tests.standalone_tests.conftest import remove_localnodetmp_rabbitmq

WAIT_TIME_LIMIT = 120


@pytest.fixture(scope="session")
def controller_config_mock():
    controller_config = AttrDict(
        {
            "log_level": "DEBUG",
            "framework_log_level": "INFO",
            "deployment_type": "LOCAL",
            "node_landscape_aggregator_update_interval": 10,
            "localnodes": {
                "config_file": "./tests/standalone_tests/testing_env_configs/test_node_landscape_aggregator.json",
                "dns": "",
                "port": "",
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
def init_background_controller_logger():
    ctrl_logger.set_background_service_logger("DEBUG")


@pytest.fixture(autouse=True, scope="session")
def patch_nodes_addresses(controller_config_mock):
    with patch(
        "mipengine.controller.nodes_addresses.controller_config", controller_config_mock
    ):
        yield


@pytest.fixture(autouse=True, scope="session")
def patch_celery_app(controller_config_mock):
    with patch(
        "mipengine.controller.celery_app.controller_config", controller_config_mock
    ):
        yield


@pytest.mark.skip(reason="https://team-1617704806227.atlassian.net/browse/MIP-625")
@pytest.mark.slow
def test_update_loop_node_service_down(
    load_data_localnodetmp,
    globalnode_node_service,
    localnodetmp_node_service,
    reset_node_landscape_aggregator,
):
    localnodetmp_node_id = get_localnodetmp_node_id()

    node_landscape_aggregator = NodeLandscapeAggregator()
    node_landscape_aggregator.start()

    # wait until node registry gets the nodes info
    start = time.time()
    while (
        not localnodetmp_node_id in node_landscape_aggregator.get_all_local_nodes()
        or not node_landscape_aggregator.get_cdes_per_data_model()
        or not node_landscape_aggregator.get_datasets_locations()
    ):
        if time.time() - start > WAIT_TIME_LIMIT:
            pytest.fail(
                f"Node registry did not contain the tmplocalnode during {WAIT_TIME_LIMIT=}"
            )
        time.sleep(1)

    data_models = node_landscape_aggregator.get_cdes_per_data_model()
    assert "tbi:0.1" and "dementia:0.1" in data_models
    assert (
        len(data_models["tbi:0.1"].values) == 21
        and len(data_models["dementia:0.1"].values) == 186
    )

    kill_service(localnodetmp_node_service)

    # wait until node registry removes tmplocalnode
    start = time.time()
    while localnodetmp_node_id in node_landscape_aggregator.get_all_local_nodes():
        if time.time() - start > WAIT_TIME_LIMIT:
            pytest.fail(
                f"Node registry did not remove the tmplocalnode during {WAIT_TIME_LIMIT=}"
            )
        time.sleep(1)

    # restart tmplocalnode node service (the celery app)
    localnodetmp_node_service_proc = start_localnodetmp_node_service()

    # wait until node registry re-added tmplocalnode
    start = time.time()
    while localnodetmp_node_id not in node_landscape_aggregator.get_all_local_nodes():
        if time.time() - start > WAIT_TIME_LIMIT:
            pytest.fail(
                f"Node registry did not contain the tmplocalnode during {WAIT_TIME_LIMIT=}"
            )
        time.sleep(1)

    data_models = node_landscape_aggregator.get_cdes_per_data_model()
    assert "tbi:0.1" and "dementia:0.1" in data_models
    assert (
        len(data_models["tbi:0.1"].values) == 21
        and len(data_models["dementia:0.1"].values) == 186
    )

    node_landscape_aggregator.stop()

    # the node service was started in here, so it must manually be killed, otherwise it is
    # alive through the whole pytest session and is erroneously accessed by other tests
    # where the node service is supposedly down
    kill_service(localnodetmp_node_service_proc)


@pytest.mark.skip(reason="https://team-1617704806227.atlassian.net/browse/MIP-625")
@pytest.mark.slow
def test_update_loop_rabbitmq_down(
    load_data_localnodetmp,
    globalnode_node_service,
    localnodetmp_node_service,
    reset_node_landscape_aggregator,
):
    # get tmp localnode node_id from config file
    localnodetmp_node_id = get_localnodetmp_node_id()

    node_landscape_aggregator = NodeLandscapeAggregator()
    node_landscape_aggregator.start()

    # wait until node registry and data model registry get the localnodetmp info
    start = time.time()
    while (
        not localnodetmp_node_id in node_landscape_aggregator.get_all_local_nodes()
        or not node_landscape_aggregator.get_cdes_per_data_model()
        or not node_landscape_aggregator.get_datasets_locations()
    ):
        if time.time() - start > WAIT_TIME_LIMIT:
            pytest.fail(
                f"Node registry did not contain the tmplocalnode during {WAIT_TIME_LIMIT=}"
            )
        time.sleep(1)

    data_models = node_landscape_aggregator.get_cdes_per_data_model()
    assert "tbi:0.1" and "dementia:0.1" in data_models
    assert (
        len(data_models["tbi:0.1"].values) == 21
        and len(data_models["dementia:0.1"].values) == 186
    )

    remove_localnodetmp_rabbitmq()

    # wait until node registry no longer contains tmplocalnode
    while localnodetmp_node_id in node_landscape_aggregator.get_all_local_nodes():
        if time.time() - start > WAIT_TIME_LIMIT:
            pytest.fail(
                f"Node registry did not remove the tmplocalnode during {WAIT_TIME_LIMIT=}"
            )
        time.sleep(1)

    # restart tmplocalnode rabbitmq container
    _create_rabbitmq_container(RABBITMQ_LOCALNODETMP_NAME, RABBITMQ_LOCALNODETMP_PORT)
    localnodetmp_node_service_proc = start_localnodetmp_node_service()

    # wait until node registry contains tmplocalnode
    start = time.time()
    while localnodetmp_node_id not in node_landscape_aggregator.get_all_local_nodes():
        if time.time() - start > WAIT_TIME_LIMIT:
            pytest.fail(
                f"Node registry did not add the tmplocalnode during {WAIT_TIME_LIMIT=}"
            )
        time.sleep(1)

    data_models = node_landscape_aggregator.get_cdes_per_data_model()
    assert "tbi:0.1" and "dementia:0.1" in data_models
    assert (
        len(data_models["tbi:0.1"].values) == 21
        and len(data_models["dementia:0.1"].values) == 186
    )

    node_landscape_aggregator.stop()

    # the node service was started in here, so it must manually be killed, otherwise it is
    # alive through the whole pytest session and is erroneously accessed by other tests
    # where the node service is supposedly down
    kill_service(localnodetmp_node_service)


@pytest.mark.skip(reason="https://team-1617704806227.atlassian.net/browse/MIP-625")
@pytest.mark.slow
def test_update_loop_data_models_removed(
    load_data_localnodetmp,
    globalnode_node_service,
    localnodetmp_node_service,
    reset_node_landscape_aggregator,
):

    # get tmp localnode node_id from config file
    localnodetmp_node_id = get_localnodetmp_node_id()

    node_landscape_aggregator = NodeLandscapeAggregator()
    node_landscape_aggregator.start()

    # wait until node registry and data model registry get the localnodetmp info
    start = time.time()
    while (
        not localnodetmp_node_id in node_landscape_aggregator.get_all_local_nodes()
        or not node_landscape_aggregator.get_cdes_per_data_model()
        or not node_landscape_aggregator.get_datasets_locations()
    ):
        if time.time() - start > WAIT_TIME_LIMIT:
            pytest.fail(
                f"Node registry did not contain the tmplocalnode during {WAIT_TIME_LIMIT=}"
            )
        # await asyncio.sleep(2)
        time.sleep(1)

    data_models = node_landscape_aggregator.get_cdes_per_data_model()
    assert "tbi:0.1" and "dementia:0.1" in data_models
    assert (
        len(data_models["tbi:0.1"].values) == 21
        and len(data_models["dementia:0.1"].values) == 186
    )

    remove_data_model_from_localnodetmp_monetdb("dementia:0.1")

    # Wait until data model registry no longer contains 'dementia:0.1' and that the tmp node is there.
    # If NLA updates when the data model is being removed from the db it will crash and remove the node completely.
    # We need to wait for it to be added again.
    start = time.time()
    while "dementia:0.1" in node_landscape_aggregator.get_cdes_per_data_model():
        if time.time() - start > WAIT_TIME_LIMIT:
            pytest.fail(
                f"Data model registry did not remove the 'dementia:0.1' during {WAIT_TIME_LIMIT=}"
            )
        time.sleep(1)

    data_models = node_landscape_aggregator.get_cdes_per_data_model()

    assert "tbi:0.1" in node_landscape_aggregator.get_cdes_per_data_model()
    assert (
        len(node_landscape_aggregator.get_cdes_per_data_model()["tbi:0.1"].values) == 21
    )

    remove_data_model_from_localnodetmp_monetdb("tbi:0.1")
    # Wait until data model registry no longer contains 'tbi:0.1' and that the tmp node is there.
    # If NLA updates when the data model is being removed from the db it will crash and remove the node completely.
    # We need to wait for it to be added again.
    start = time.time()
    while "tbi:0.1" in node_landscape_aggregator.get_cdes_per_data_model():
        if time.time() - start > WAIT_TIME_LIMIT:
            pytest.fail(
                f"Data model registry did not remove the 'tbi:0.1' during {WAIT_TIME_LIMIT=}"
            )
        time.sleep(1)

    # Reload the data models in the db
    local_nodes = node_landscape_aggregator.get_all_local_nodes()
    _load_data_monetdb_container(
        db_ip=local_nodes[localnodetmp_node_id].db_ip,
        db_port=local_nodes[localnodetmp_node_id].db_port,
    )

    # wait until data models are re-loaded in the data model registry
    start = time.time()
    while (
        "dementia:0.1" not in node_landscape_aggregator.get_cdes_per_data_model()
        or "tbi:0.1" not in node_landscape_aggregator.get_cdes_per_data_model()
    ):
        if time.time() - start > WAIT_TIME_LIMIT:
            pytest.fail(
                f"NLA did not contain the data models 'tbi:0.1' and 'dementia:0.1' during {WAIT_TIME_LIMIT=}"
            )
        time.sleep(1)

    data_models = node_landscape_aggregator.get_cdes_per_data_model()
    assert "tbi:0.1" in data_models and "dementia:0.1" in data_models
    assert (
        len(data_models["tbi:0.1"].values) == 21
        and len(data_models["dementia:0.1"].values) == 186
    )

    node_landscape_aggregator.stop()

    # the node service was started in here, so it must manually be killed, otherwise it is
    # alive through the whole pytest session and is erroneously accessed by other tests
    # where the node service is supposedly down
    kill_service(localnodetmp_node_service)


def remove_data_model_from_localnodetmp_monetdb(data_model):
    data_model_code, data_model_version = data_model.split(":")
    _remove_data_model_from_localnodetmp_monetdb(
        data_model_code=data_model_code,
        data_model_version=data_model_version,
    )


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

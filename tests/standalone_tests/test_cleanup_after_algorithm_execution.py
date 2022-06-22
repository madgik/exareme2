import asyncio
import time
from copy import deepcopy
from os import path
from unittest.mock import patch

import pytest

from mipengine import AttrDict
from mipengine.controller import controller_logger as ctrl_logger
from mipengine.controller.algorithm_execution_DTOs import NodesTasksHandlersDTO
from mipengine.controller.api.algorithm_request_dto import AlgorithmInputDataDTO
from mipengine.controller.api.algorithm_request_dto import AlgorithmRequestDTO
from mipengine.controller.controller import Controller
from mipengine.controller.controller import get_a_uniqueid
from tests.standalone_tests.conftest import ALGORITHM_FOLDERS_ENV_VARIABLE_VALUE
from tests.standalone_tests.conftest import CONTROLLER_LOCALNODES_CONFIG_FILE
from tests.standalone_tests.conftest import LOCALNODETMP_CONFIG_FILE
from tests.standalone_tests.conftest import RABBITMQ_LOCALNODETMP_NAME
from tests.standalone_tests.conftest import RABBITMQ_LOCALNODETMP_PORT
from tests.standalone_tests.conftest import TEST_ENV_CONFIG_FOLDER
from tests.standalone_tests.conftest import _create_node_service
from tests.standalone_tests.conftest import _create_rabbitmq_container
from tests.standalone_tests.conftest import create_node_tasks_handler_celery
from tests.standalone_tests.conftest import kill_service
from tests.standalone_tests.conftest import remove_localnodetmp_rabbitmq

WAIT_CLEANUP_TIME_LIMIT = 40
WAIT_BEFORE_BRING_TMPNODE_DOWN = 20
WAIT_BACKGROUND_TASKS_TO_FINISH = 30
NLA_WAIT_TIME_LIMIT = 60


@pytest.fixture(scope="session")
def controller_config_dict_mock():
    controller_config = {
        "log_level": "DEBUG",
        "framework_log_level": "INFO",
        "deployment_type": "LOCAL",
        "node_landscape_aggregator_update_interval": 2,
        "cleanup": {
            "contextids_cleanup_folder": "/tmp",
            "nodes_cleanup_interval": 2,
            "contextid_release_timelimit": 3600,  # 1hour
        },
        "localnodes": {
            "config_file": "./tests/standalone_tests/testing_env_configs/"
            + CONTROLLER_LOCALNODES_CONFIG_FILE,
            "dns": "",
            "port": "",
        },
        "rabbitmq": {
            "user": "user",
            "password": "password",
            "vhost": "user_vhost",
            "celery_tasks_timeout": 40,
            "celery_run_udf_task_timeout": 40,
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

    return controller_config


@pytest.fixture(autouse=True, scope="session")
def patch_controller(controller_config_dict_mock):
    with patch(
        "mipengine.controller.controller.controller_config",
        AttrDict(controller_config_dict_mock),
    ):
        yield


@pytest.fixture(autouse=True, scope="session")
def patch_cleaner(controller_config_dict_mock):
    with patch(
        "mipengine.controller.cleaner.controller_config",
        AttrDict(controller_config_dict_mock),
    ):
        yield


@pytest.fixture(scope="function")
def patch_cleaner_small_release_timelimit(controller_config_dict_mock):
    controller_config_dict_mock_copy = deepcopy(controller_config_dict_mock)
    controller_config_dict_mock_copy["cleanup"]["contextid_release_timelimit"] = 5
    with patch(
        "mipengine.controller.cleaner.controller_config",
        AttrDict(controller_config_dict_mock_copy),
    ):
        yield


@pytest.fixture(autouse=True, scope="session")
def patch_node_landscape_aggregator(controller_config_dict_mock):
    with patch(
        "mipengine.controller.node_landscape_aggregator.controller_config",
        AttrDict(controller_config_dict_mock),
    ), patch(
        "mipengine.controller.node_landscape_aggregator.NODE_LANDSCAPE_AGGREGATOR_UPDATE_INTERVAL",
        AttrDict(controller_config_dict_mock).node_landscape_aggregator_update_interval,
    ), patch(
        "mipengine.controller.node_landscape_aggregator.CELERY_TASKS_TIMEOUT",
        AttrDict(controller_config_dict_mock).rabbitmq.celery_tasks_timeout,
    ):
        yield


@pytest.fixture(autouse=True, scope="session")
def patch_algorithm_executor(controller_config_dict_mock):
    with patch(
        "mipengine.controller.algorithm_executor.ctrl_config",
        AttrDict(controller_config_dict_mock),
    ):
        yield


@pytest.fixture(autouse=True, scope="session")
def patch_nodes_addresses(controller_config_dict_mock):
    with patch(
        "mipengine.controller.nodes_addresses.controller_config",
        AttrDict(controller_config_dict_mock),
    ):
        yield


@pytest.fixture(autouse=True, scope="session")
def patch_celery_app(controller_config_dict_mock):
    with patch(
        "mipengine.controller.celery_app.controller_config",
        AttrDict(controller_config_dict_mock),
    ):
        yield


algorithm_request_dto = AlgorithmRequestDTO(
    inputdata=AlgorithmInputDataDTO(
        data_model="dementia:0.1",
        datasets=[
            "edsd0",
            "edsd1",
            "edsd2",
            "edsd3",
            "edsd4",
            "edsd5",
            "edsd6",
            "edsd7",
            "edsd8",
            "edsd9",
        ],
        filters={
            "condition": "AND",
            "rules": [
                {
                    "id": "dataset",
                    "type": "string",
                    "value": [
                        "edsd0",
                        "edsd1",
                        "edsd2",
                        "edsd3",
                        "edsd4",
                        "edsd5",
                        "edsd6",
                        "edsd7",
                        "edsd8",
                        "edsd9",
                    ],
                    "operator": "in",
                },
                {
                    "condition": "AND",
                    "rules": [
                        {
                            "id": variable,
                            "type": "string",
                            "operator": "is_not_null",
                            "value": None,
                        }
                        for variable in [
                            "lefthippocampus",
                            "righthippocampus",
                            "rightppplanumpolare",
                            "leftamygdala",
                            "rightamygdala",
                            "alzheimerbroadcategory",
                        ]
                    ],
                },
            ],
            "valid": True,
        },
        x=[
            "lefthippocampus",
            "righthippocampus",
            "rightppplanumpolare",
            "leftamygdala",
            "rightamygdala",
        ],
        y=["alzheimerbroadcategory"],
    ),
    parameters={"classes": ["AD", "CN"]},
)


@pytest.mark.slow
@pytest.mark.asyncio
async def test_cleanup_after_uninterrupted_algorithm_execution(
    load_data_localnode1,
    load_data_localnode2,
    globalnode_tasks_handler,
    localnode1_tasks_handler,
    localnode2_tasks_handler,
    reset_node_landscape_aggregator,
):

    controller = Controller()

    # start node landscape aggregator
    # node landscape aggregator has to run on the background because it is used by the Cleaner
    controller.start_node_landscape_aggregator()
    # wait until node registry gets the nodes info
    start = time.time()
    while not controller._node_landscape_aggregator._node_registry._nodes:
        if time.time() - start > NLA_WAIT_TIME_LIMIT:
            pytest.fail(
                "Exceeded max retries while waiting for the node registry to contain the tmplocalnode"
            )
        await asyncio.sleep(2)

    # Start the cleanup loop
    controller._cleaner._reset_cleanup()
    controller.start_cleanup_loop()
    request_id = get_a_uniqueid()
    context_id = get_a_uniqueid()
    algorithm_name = "logistic_regression"
    algo_execution_logger = ctrl_logger.get_request_logger(request_id=request_id)

    controller._cleaner.add_contextid_for_cleanup(
        context_id,
        [
            globalnode_tasks_handler.node_id,
            localnode1_tasks_handler.node_id,
            localnode2_tasks_handler.node_id,
        ],
    )

    try:
        await controller._exec_algorithm_with_task_handlers(
            request_id=request_id,
            context_id=context_id,
            algorithm_name=algorithm_name,
            algorithm_request_dto=algorithm_request_dto,
            datasets_per_local_node={
                localnode1_tasks_handler.node_id: [
                    "edsd0",
                    "edsd1",
                    "edsd2",
                    "edsd3",
                    "edsd4",
                    "edsd5",
                    "edsd6",
                    "edsd7",
                    "edsd8",
                    "edsd9",
                ],
                localnode2_tasks_handler.node_id: [
                    "edsd0",
                    "edsd1",
                    "edsd2",
                    "edsd3",
                    "edsd4",
                    "edsd5",
                    "edsd6",
                    "edsd7",
                    "edsd8",
                    "edsd9",
                ],
            },
            tasks_handlers=NodesTasksHandlersDTO(
                global_node_tasks_handler=globalnode_tasks_handler,
                local_nodes_tasks_handlers=[
                    localnode1_tasks_handler,
                    localnode2_tasks_handler,
                ],
            ),
            logger=algo_execution_logger,
        )
    except Exception as exc:
        pytest.fail(f"Execution of the algorithm failed with {exc=}")

    globalnode_tables_before_cleanup = globalnode_tasks_handler.get_tables(
        request_id=request_id, context_id=context_id
    )
    localnode1_tables_before_cleanup = localnode1_tasks_handler.get_tables(
        request_id=request_id, context_id=context_id
    )
    localnode2_tables_before_cleanup = localnode2_tasks_handler.get_tables(
        request_id=request_id, context_id=context_id
    )

    controller._cleaner.release_contextid_for_cleanup(context_id=context_id)

    globalnode_tables_after_cleanup = globalnode_tasks_handler.get_tables(
        request_id=request_id, context_id=context_id
    )
    localnode1_tables_after_cleanup = localnode1_tasks_handler.get_tables(
        request_id=request_id, context_id=context_id
    )
    localnode2_tables_after_cleanup = localnode2_tasks_handler.get_tables(
        request_id=request_id, context_id=context_id
    )

    start = time.time()
    while (
        globalnode_tables_after_cleanup
        and localnode1_tables_after_cleanup
        and localnode2_tables_after_cleanup
    ):
        globalnode_tables_after_cleanup = globalnode_tasks_handler.get_tables(
            request_id=request_id, context_id=context_id
        )
        localnode1_tables_after_cleanup = localnode1_tasks_handler.get_tables(
            request_id=request_id, context_id=context_id
        )
        localnode2_tables_after_cleanup = localnode2_tasks_handler.get_tables(
            request_id=request_id, context_id=context_id
        )

        now = time.time()
        if now - start > WAIT_CLEANUP_TIME_LIMIT:
            pytest.fail(
                f"Some of the nodes were not cleaned during {WAIT_CLEANUP_TIME_LIMIT=}"
            )
        await asyncio.sleep(2)

    controller.stop_node_landscape_aggregator()
    controller.stop_cleanup_loop()
    # give some time for node landscape aggregator and cleanup background tasks to stop gracefully
    await asyncio.sleep(WAIT_BACKGROUND_TASKS_TO_FINISH)

    if (
        globalnode_tables_before_cleanup
        and not globalnode_tables_after_cleanup
        and localnode1_tables_before_cleanup
        and not localnode1_tables_after_cleanup
        and localnode2_tables_before_cleanup
        and not localnode2_tables_after_cleanup
    ):
        assert True
    else:
        assert False


@pytest.mark.slow
@pytest.mark.asyncio
async def test_cleanup_after_uninterrupted_algorithm_execution_without_releasing_contextid(
    patch_cleaner_small_release_timelimit,
    load_data_localnode1,
    load_data_localnode2,
    globalnode_tasks_handler,
    localnode1_tasks_handler,
    localnode2_tasks_handler,
    reset_node_landscape_aggregator,
):

    controller = Controller()

    # start node landscape aggregator
    # node landscape aggregator has to run on the background because it is used by the Cleaner
    controller.start_node_landscape_aggregator()
    # wait until node registry gets the nodes info
    start = time.time()
    while not controller._node_landscape_aggregator._node_registry._nodes:
        if time.time() - start > NLA_WAIT_TIME_LIMIT:
            pytest.fail(
                "Exceeded max retries while waiting for the node registry to contain the tmplocalnode"
            )
        await asyncio.sleep(2)

    # Start the cleanup loop
    controller._cleaner._reset_cleanup()
    controller.start_cleanup_loop()

    request_id = get_a_uniqueid()
    context_id = get_a_uniqueid()
    algorithm_name = "logistic_regression"
    algo_execution_logger = ctrl_logger.get_request_logger(request_id=request_id)

    try:
        await controller._exec_algorithm_with_task_handlers(
            request_id=request_id,
            context_id=context_id,
            algorithm_name=algorithm_name,
            algorithm_request_dto=algorithm_request_dto,
            datasets_per_local_node={
                localnode1_tasks_handler.node_id: [
                    "edsd0",
                    "edsd1",
                    "edsd2",
                    "edsd3",
                    "edsd4",
                    "edsd5",
                    "edsd6",
                    "edsd7",
                    "edsd8",
                    "edsd9",
                ],
                localnode2_tasks_handler.node_id: [
                    "edsd0",
                    "edsd1",
                    "edsd2",
                    "edsd3",
                    "edsd4",
                    "edsd5",
                    "edsd6",
                    "edsd7",
                    "edsd8",
                    "edsd9",
                ],
            },
            tasks_handlers=NodesTasksHandlersDTO(
                global_node_tasks_handler=globalnode_tasks_handler,
                local_nodes_tasks_handlers=[
                    localnode1_tasks_handler,
                    localnode2_tasks_handler,
                ],
            ),
            logger=algo_execution_logger,
        )
    except Exception as exc:
        pytest.fail(f"Execution of the algorithm failed with {exc=}")

    globalnode_tables_before_cleanup = globalnode_tasks_handler.get_tables(
        request_id=request_id, context_id=context_id
    )
    localnode1_tables_before_cleanup = localnode1_tasks_handler.get_tables(
        request_id=request_id, context_id=context_id
    )
    localnode2_tables_before_cleanup = localnode2_tasks_handler.get_tables(
        request_id=request_id, context_id=context_id
    )

    controller._cleaner.add_contextid_for_cleanup(
        context_id,
        [
            globalnode_tasks_handler.node_id,
            localnode1_tasks_handler.node_id,
            localnode2_tasks_handler.node_id,
        ],
    )
    globalnode_tables_after_cleanup = globalnode_tasks_handler.get_tables(
        request_id=request_id, context_id=context_id
    )
    localnode1_tables_after_cleanup = localnode1_tasks_handler.get_tables(
        request_id=request_id, context_id=context_id
    )
    localnode2_tables_after_cleanup = localnode2_tasks_handler.get_tables(
        request_id=request_id, context_id=context_id
    )

    start = time.time()
    while (
        globalnode_tables_after_cleanup
        and localnode1_tables_after_cleanup
        and localnode2_tables_after_cleanup
    ):
        globalnode_tables_after_cleanup = globalnode_tasks_handler.get_tables(
            request_id=request_id, context_id=context_id
        )
        localnode1_tables_after_cleanup = localnode1_tasks_handler.get_tables(
            request_id=request_id, context_id=context_id
        )
        localnode2_tables_after_cleanup = localnode2_tasks_handler.get_tables(
            request_id=request_id, context_id=context_id
        )
        now = time.time()
        if now - start > WAIT_CLEANUP_TIME_LIMIT:
            pytest.fail(
                f"Some of the nodes were not cleaned during {WAIT_CLEANUP_TIME_LIMIT=}"
            )
        await asyncio.sleep(2)

    controller.stop_node_landscape_aggregator()
    controller.stop_cleanup_loop()
    # give some time for node landscape aggregator and cleanup background tasks to stop gracefully
    await asyncio.sleep(WAIT_BACKGROUND_TASKS_TO_FINISH)

    if (
        globalnode_tables_before_cleanup
        and not globalnode_tables_after_cleanup
        and localnode1_tables_before_cleanup
        and not localnode1_tables_after_cleanup
        and localnode2_tables_before_cleanup
        and not localnode2_tables_after_cleanup
    ):
        assert True
    else:
        assert False


@pytest.mark.slow
@pytest.mark.asyncio
async def test_cleanup_rabbitmq_down_algorithm_execution(
    load_data_localnode1,
    load_data_localnodetmp,
    globalnode_tasks_handler,
    localnode1_tasks_handler,
    localnodetmp_tasks_handler,
    localnodetmp_node_service,
    reset_node_landscape_aggregator,
):
    controller = Controller()

    # start node landscape aggregator
    # node landscape aggregator has to run on the background because it is used by the Cleaner
    controller.start_node_landscape_aggregator()
    # wait until node registry gets the nodes info
    start = time.time()
    while not controller._node_landscape_aggregator._node_registry._nodes:
        if time.time() - start > NLA_WAIT_TIME_LIMIT:
            pytest.fail(
                "Exceeded max retries while waiting for the node registry to contain the tmplocalnode"
            )
        await asyncio.sleep(2)

    # Start the cleanup loop
    controller._cleaner._reset_cleanup()
    controller.start_cleanup_loop()

    request_id = get_a_uniqueid()
    context_id = get_a_uniqueid()
    algorithm_name = "logistic_regression"
    algo_execution_logger = ctrl_logger.get_request_logger(request_id=request_id)

    controller._cleaner.add_contextid_for_cleanup(
        context_id,
        [
            globalnode_tasks_handler.node_id,
            localnode1_tasks_handler.node_id,
            localnodetmp_tasks_handler.node_id,
        ],
    )

    # start executing the algorithm but do not wait for it
    asyncio.create_task(
        controller._exec_algorithm_with_task_handlers(
            request_id=request_id,
            context_id=context_id,
            algorithm_name=algorithm_name,
            algorithm_request_dto=algorithm_request_dto,
            datasets_per_local_node={
                localnode1_tasks_handler.node_id: [
                    "edsd0",
                    "edsd1",
                    "edsd2",
                    "edsd3",
                    "edsd4",
                    "edsd5",
                    "edsd6",
                    "edsd7",
                    "edsd8",
                    "edsd9",
                ],
                localnodetmp_tasks_handler.node_id: [
                    "edsd0",
                    "edsd1",
                    "edsd2",
                    "edsd3",
                    "edsd4",
                    "edsd5",
                    "edsd6",
                    "edsd7",
                    "edsd8",
                    "edsd9",
                ],
            },
            tasks_handlers=NodesTasksHandlersDTO(
                global_node_tasks_handler=globalnode_tasks_handler,
                local_nodes_tasks_handlers=[
                    localnode1_tasks_handler,
                    localnodetmp_tasks_handler,
                ],
            ),
            logger=algo_execution_logger,
        )
    )

    await asyncio.sleep(WAIT_BEFORE_BRING_TMPNODE_DOWN)

    globalnode_tables_before_cleanup = globalnode_tasks_handler.get_tables(
        request_id=request_id, context_id=context_id
    )
    localnode1_tables_before_cleanup = localnode1_tasks_handler.get_tables(
        request_id=request_id, context_id=context_id
    )
    localnodetmp_tables_before_cleanup = localnodetmp_tasks_handler.get_tables(
        request_id=request_id, context_id=context_id
    )

    remove_localnodetmp_rabbitmq()
    kill_service(localnodetmp_node_service)

    controller._cleaner.release_contextid_for_cleanup(context_id=context_id)

    # restart tmplocalnode rabbitmq container
    _create_rabbitmq_container(RABBITMQ_LOCALNODETMP_NAME, RABBITMQ_LOCALNODETMP_PORT)
    localnodetmp_node_service_proc = start_localnodetmp_node_service()
    localnodetmp_tasks_handler = create_node_tasks_handler_celery(
        path.join(TEST_ENV_CONFIG_FOLDER, LOCALNODETMP_CONFIG_FILE)
    )

    globalnode_tables_after_cleanup = globalnode_tasks_handler.get_tables(
        request_id=request_id, context_id=context_id
    )
    localnode1_tables_after_cleanup = localnode1_tasks_handler.get_tables(
        request_id=request_id, context_id=context_id
    )

    localnodetmp_tables_after_cleanup = localnodetmp_tasks_handler.get_tables(
        request_id=request_id, context_id=context_id
    )

    start = time.time()
    while (
        globalnode_tables_after_cleanup
        or localnode1_tables_after_cleanup
        or localnodetmp_tables_after_cleanup
    ):
        globalnode_tables_after_cleanup = globalnode_tasks_handler.get_tables(
            request_id=request_id, context_id=context_id
        )
        localnode1_tables_after_cleanup = localnode1_tasks_handler.get_tables(
            request_id=request_id, context_id=context_id
        )
        localnodetmp_tables_after_cleanup = localnodetmp_tasks_handler.get_tables(
            request_id=request_id, context_id=context_id
        )

        now = time.time()
        if now - start > WAIT_CLEANUP_TIME_LIMIT:
            pytest.fail(
                f"Some of the nodes were not cleaned during {WAIT_CLEANUP_TIME_LIMIT=}"
            )

        await asyncio.sleep(2)

    controller.stop_node_landscape_aggregator()
    controller.stop_cleanup_loop()
    localnodetmp_tasks_handler.close()

    # give some time for node landscape aggregator and cleanup background tasks to finish gracefully
    await asyncio.sleep(WAIT_BACKGROUND_TASKS_TO_FINISH)

    # the node service was started in here so it must manually killed, otherwise it is
    # alive through the whole pytest session and is erroneously accessed by other tests
    # where teh node service is supposedly down
    kill_service(localnodetmp_node_service_proc)

    if (
        globalnode_tables_before_cleanup
        and not globalnode_tables_after_cleanup
        and localnode1_tables_before_cleanup
        and not localnode1_tables_after_cleanup
        and localnodetmp_tables_before_cleanup
        and not localnodetmp_tables_after_cleanup
    ):
        assert True
    else:
        assert False


@pytest.mark.slow
@pytest.mark.asyncio
async def test_cleanup_node_service_down_algorithm_execution(
    init_data_globalnode,
    load_data_localnode1,
    load_data_localnodetmp,
    globalnode_tasks_handler,
    localnode1_tasks_handler,
    localnodetmp_tasks_handler,
    localnodetmp_node_service,
    reset_node_landscape_aggregator,
):
    controller = Controller()

    # start node landscape aggregator
    # node landscape aggregator has to run on the background because it is used by the Cleaner
    controller.start_node_landscape_aggregator()
    # wait until node registry gets the nodes info
    start = time.time()
    while not controller._node_landscape_aggregator._node_registry._nodes:
        if time.time() - start > NLA_WAIT_TIME_LIMIT:
            pytest.fail(
                "Exceeded max retries while waiting for the node registry to contain the tmplocalnode"
            )
        await asyncio.sleep(2)

    # Start the cleanup loop
    controller._cleaner._reset_cleanup()
    controller.start_cleanup_loop()

    request_id = get_a_uniqueid()
    context_id = get_a_uniqueid()
    algorithm_name = "logistic_regression"
    algo_execution_logger = ctrl_logger.get_request_logger(request_id=request_id)

    controller._cleaner.add_contextid_for_cleanup(
        context_id,
        [
            globalnode_tasks_handler.node_id,
            localnode1_tasks_handler.node_id,
            localnodetmp_tasks_handler.node_id,
        ],
    )

    # start executing the algorithm but do not wait for it
    asyncio.create_task(
        controller._exec_algorithm_with_task_handlers(
            request_id=request_id,
            context_id=context_id,
            algorithm_name=algorithm_name,
            algorithm_request_dto=algorithm_request_dto,
            datasets_per_local_node={
                localnode1_tasks_handler.node_id: [
                    "edsd0",
                    "edsd1",
                    "edsd2",
                    "edsd3",
                    "edsd4",
                    "edsd5",
                    "edsd6",
                    "edsd7",
                    "edsd8",
                    "edsd9",
                ],
                localnodetmp_tasks_handler.node_id: [
                    "edsd0",
                    "edsd1",
                    "edsd2",
                    "edsd3",
                    "edsd4",
                    "edsd5",
                    "edsd6",
                    "edsd7",
                    "edsd8",
                    "edsd9",
                ],
            },
            tasks_handlers=NodesTasksHandlersDTO(
                global_node_tasks_handler=globalnode_tasks_handler,
                local_nodes_tasks_handlers=[
                    localnode1_tasks_handler,
                    localnodetmp_tasks_handler,
                ],
            ),
            logger=algo_execution_logger,
        )
    )

    await asyncio.sleep(WAIT_BEFORE_BRING_TMPNODE_DOWN)

    globalnode_tables_before_cleanup = globalnode_tasks_handler.get_tables(
        request_id=request_id, context_id=context_id
    )
    localnode1_tables_before_cleanup = localnode1_tasks_handler.get_tables(
        request_id=request_id, context_id=context_id
    )
    localnodetmp_tables_before_cleanup = localnodetmp_tasks_handler.get_tables(
        request_id=request_id, context_id=context_id
    )

    kill_service(localnodetmp_node_service)

    controller._cleaner.release_contextid_for_cleanup(context_id=context_id)

    # restart tmplocalnode node service (the celery app)
    localnodetmp_node_service_proc = start_localnodetmp_node_service()

    globalnode_tables_after_cleanup = globalnode_tasks_handler.get_tables(
        request_id=request_id, context_id=context_id
    )
    localnode1_tables_after_cleanup = localnode1_tasks_handler.get_tables(
        request_id=request_id, context_id=context_id
    )
    localnodetmp_tables_after_cleanup = localnodetmp_tasks_handler.get_tables(
        request_id=request_id, context_id=context_id
    )

    start = time.time()
    while (
        globalnode_tables_after_cleanup
        and localnode1_tables_after_cleanup
        and localnodetmp_tables_after_cleanup
    ):
        globalnode_tables_after_cleanup = globalnode_tasks_handler.get_tables(
            request_id=request_id, context_id=context_id
        )
        localnode1_tables_after_cleanup = localnode1_tasks_handler.get_tables(
            request_id=request_id, context_id=context_id
        )
        localnodetmp_tables_after_cleanup = localnodetmp_tasks_handler.get_tables(
            request_id=request_id, context_id=context_id
        )
        now = time.time()
        if now - start > WAIT_CLEANUP_TIME_LIMIT:
            pytest.fail(
                f"Some of the nodes were not cleaned during {WAIT_CLEANUP_TIME_LIMIT=}"
            )
        await asyncio.sleep(2)

    await asyncio.sleep(2)

    controller.stop_node_landscape_aggregator()
    controller.stop_cleanup_loop()

    # give some time for node landscape aggregator and cleanup background tasks to finish gracefully
    await asyncio.sleep(WAIT_BACKGROUND_TASKS_TO_FINISH)

    # the node service was started in here so it must manually killed, otherwise it is
    # alive through the whole pytest session and is erroneously accessed by other tests
    # where teh node service is supposedly down
    kill_service(localnodetmp_node_service_proc)

    if (
        globalnode_tables_before_cleanup
        and not globalnode_tables_after_cleanup
        and localnode1_tables_before_cleanup
        and not localnode1_tables_after_cleanup
        and localnodetmp_tables_before_cleanup
        and not localnodetmp_tables_after_cleanup
    ):
        assert True
    else:
        assert False


@pytest.mark.slow
@pytest.mark.asyncio
async def test_cleanup_controller_restart(
    init_data_globalnode,
    load_data_localnode1,
    load_data_localnodetmp,
    globalnode_tasks_handler,
    localnode1_tasks_handler,
    localnodetmp_tasks_handler,
    reset_node_landscape_aggregator,
):
    controller = Controller()

    # start node landscape aggregator
    # node landscape aggregator has to run on the background because it is used by the Cleaner
    controller.start_node_landscape_aggregator()
    # wait until node registry gets the nodes info
    start = time.time()
    while not controller._node_landscape_aggregator._node_registry._nodes:
        if time.time() - start > NLA_WAIT_TIME_LIMIT:
            pytest.fail(
                "Exceeded max retries while waiting for the node registry to contain the tmplocalnode"
            )
        await asyncio.sleep(2)

    # Start the cleanup loop
    controller._cleaner._reset_cleanup()
    controller.start_cleanup_loop()

    request_id = get_a_uniqueid()
    context_id = get_a_uniqueid()
    algorithm_name = "logistic_regression"
    algo_execution_logger = ctrl_logger.get_request_logger(request_id=request_id)

    controller._cleaner.add_contextid_for_cleanup(
        context_id,
        [
            globalnode_tasks_handler.node_id,
            localnode1_tasks_handler.node_id,
            localnodetmp_tasks_handler.node_id,
        ],
    )
    # start executing the algorithm but do not wait for it
    asyncio.create_task(
        controller._exec_algorithm_with_task_handlers(
            request_id=request_id,
            context_id=context_id,
            algorithm_name=algorithm_name,
            algorithm_request_dto=algorithm_request_dto,
            datasets_per_local_node={
                localnode1_tasks_handler.node_id: [
                    "edsd0",
                    "edsd1",
                    "edsd2",
                    "edsd3",
                    "edsd4",
                    "edsd5",
                    "edsd6",
                    "edsd7",
                    "edsd8",
                    "edsd9",
                ],
                localnodetmp_tasks_handler.node_id: [
                    "edsd0",
                    "edsd1",
                    "edsd2",
                    "edsd3",
                    "edsd4",
                    "edsd5",
                    "edsd6",
                    "edsd7",
                    "edsd8",
                    "edsd9",
                ],
            },
            tasks_handlers=NodesTasksHandlersDTO(
                global_node_tasks_handler=globalnode_tasks_handler,
                local_nodes_tasks_handlers=[
                    localnode1_tasks_handler,
                    localnodetmp_tasks_handler,
                ],
            ),
            logger=algo_execution_logger,
        )
    )

    await asyncio.sleep(WAIT_BEFORE_BRING_TMPNODE_DOWN)

    globalnode_tables_before_cleanup = globalnode_tasks_handler.get_tables(
        request_id=request_id, context_id=context_id
    )
    localnode1_tables_before_cleanup = localnode1_tasks_handler.get_tables(
        request_id=request_id, context_id=context_id
    )
    localnodetmp_tables_before_cleanup = localnodetmp_tasks_handler.get_tables(
        request_id=request_id, context_id=context_id
    )

    controller.stop_node_landscape_aggregator()
    controller.stop_cleanup_loop()
    # give some time for node landscape aggregator and cleanup background tasks to stop gracefully
    await asyncio.sleep(WAIT_BACKGROUND_TASKS_TO_FINISH)

    controller._cleaner.release_contextid_for_cleanup(context_id=context_id)

    # instantiate a new Controller
    controller = Controller()
    # start node landscape aggregator
    controller.start_node_landscape_aggregator()
    # Start the cleanup loop
    controller.start_cleanup_loop()

    globalnode_tables_after_cleanup = globalnode_tasks_handler.get_tables(
        request_id=request_id, context_id=context_id
    )
    localnode1_tables_after_cleanup = localnode1_tasks_handler.get_tables(
        request_id=request_id, context_id=context_id
    )
    localnodetmp_tables_after_cleanup = localnodetmp_tasks_handler.get_tables(
        request_id=request_id, context_id=context_id
    )

    start = time.time()
    while (
        globalnode_tables_after_cleanup
        and localnode1_tables_after_cleanup
        and localnodetmp_tables_after_cleanup
    ):
        globalnode_tables_after_cleanup = globalnode_tasks_handler.get_tables(
            request_id=request_id, context_id=context_id
        )
        localnode1_tables_after_cleanup = localnode1_tasks_handler.get_tables(
            request_id=request_id, context_id=context_id
        )
        localnodetmp_tables_after_cleanup = localnodetmp_tasks_handler.get_tables(
            request_id=request_id, context_id=context_id
        )
        now = time.time()
        if now - start > WAIT_CLEANUP_TIME_LIMIT:
            pytest.fail(
                f"Some of the nodes were not cleaned during {WAIT_CLEANUP_TIME_LIMIT=}"
            )
        await asyncio.sleep(2)

    await asyncio.sleep(2)

    controller.stop_node_landscape_aggregator()
    controller.stop_cleanup_loop()

    # give some time for node landscape aggregator and cleanup background tasks to finish gracefully
    await asyncio.sleep(WAIT_BACKGROUND_TASKS_TO_FINISH)

    if (
        globalnode_tables_before_cleanup
        and not globalnode_tables_after_cleanup
        and localnode1_tables_before_cleanup
        and not localnode1_tables_after_cleanup
        and localnodetmp_tables_before_cleanup
        and not localnodetmp_tables_after_cleanup
    ):
        assert True
    else:
        assert False


def start_localnodetmp_node_service():
    node_config_file = LOCALNODETMP_CONFIG_FILE
    algo_folders_env_variable_val = ALGORITHM_FOLDERS_ENV_VARIABLE_VALUE
    node_config_filepath = path.join(TEST_ENV_CONFIG_FOLDER, node_config_file)
    proc = _create_node_service(algo_folders_env_variable_val, node_config_filepath)
    return proc

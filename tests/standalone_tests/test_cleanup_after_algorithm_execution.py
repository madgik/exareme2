import asyncio
import json
import os
import time
from os import path
from unittest.mock import patch

import pytest
import toml

# for cdes...
from mipengine.common_data_elements import CommonDataElement
from mipengine.common_data_elements import CommonDataElements
from mipengine.common_data_elements import MetadataEnumeration
from mipengine.common_data_elements import MetadataVariable
from mipengine.controller import controller_logger as ctrl_logger
from mipengine.controller.algorithm_execution_DTOs import AlgorithmExecutionDTO
from mipengine.controller.algorithm_execution_DTOs import NodesTasksHandlersDTO
from mipengine.controller.algorithm_executor import (
    NodeUnresponsiveAlgorithmExecutionException,
)
from mipengine.controller.api.algorithm_request_dto import AlgorithmInputDataDTO
from mipengine.controller.api.algorithm_request_dto import AlgorithmRequestDTO
from mipengine.controller.controller import CLEANUP_INTERVAL
from mipengine.controller.controller import Controller
from mipengine.controller.controller import get_a_uniqueid
from mipengine.controller.node_tasks_handler_celery import NodeTasksHandlerCelery
from tests.dev_env_tests.nodes_communication import get_node_config_by_id
from tests.standalone_tests.conftest import ALGORITHM_FOLDERS_ENV_VARIABLE_VALUE
from tests.standalone_tests.conftest import LOCALNODETMP_CONFIG_FILE
from tests.standalone_tests.conftest import RABBITMQ_LOCALNODETMP_NAME
from tests.standalone_tests.conftest import RABBITMQ_LOCALNODETMP_PORT
from tests.standalone_tests.conftest import TEST_ENV_CONFIG_FOLDER
from tests.standalone_tests.conftest import _create_node_service
from tests.standalone_tests.conftest import _create_rabbitmq_container
from tests.standalone_tests.conftest import remove_localnodetmp_rabbitmq

WAIT_CLEANUP_TIME_LIMIT = 20
WAIT_BEFORE_BRING_TMPNODE_DOWN = 15
WAIT_BACKGROUND_TASKS_TO_FINISH = 20


@pytest.mark.asyncio
async def test_cleanup_after_uninterrupted_algorithm_execution(
    init_data_globalnode,
    load_data_localnode1,
    load_data_localnode2,
    globalnode_node_service,
    localnode1_node_service,
    localnode2_node_service,
):

    controller = Controller()

    # start node registry
    await controller.start_node_registry()

    # wait until node registry gets the nodes info
    while not controller.get_all_local_nodes():
        await asyncio.sleep(1)

    # Start the cleanup loop
    cleanup_task = await controller.start_cleanup_loop()

    # get all participating node tasks handlers
    nodes_tasks_handlers_dto = controller._get_nodes_tasks_handlers(
        data_model=algorithm_request_dto.inputdata.data_model,
        datasets=algorithm_request_dto.inputdata.datasets,
    )

    globalnode_tasks_handler = nodes_tasks_handlers_dto.global_node_tasks_handler
    localnode1_tasks_handler = nodes_tasks_handlers_dto.local_nodes_tasks_handlers[0]
    localnode2_tasks_handler = nodes_tasks_handlers_dto.local_nodes_tasks_handlers[1]

    request_id = get_a_uniqueid()
    context_id = get_a_uniqueid()
    algorithm_name = "logistic_regression"
    algo_execution_logger = ctrl_logger.get_request_logger(request_id=request_id)

    try:
        result = await controller._exec_algorithm_with_task_handlers(
            request_id=request_id,
            context_id=context_id,
            algorithm_name=algorithm_name,
            algorithm_request_dto=algorithm_request_dto,
            tasks_handlers=nodes_tasks_handlers_dto,
            logger=algo_execution_logger,
        )
    except:
        assert False

    globalnode_tables_before_cleanup = globalnode_tasks_handler.get_tables(
        request_id=request_id, context_id=context_id
    )
    localnode1_tables_before_cleanup = localnode1_tasks_handler.get_tables(
        request_id=request_id, context_id=context_id
    )
    localnode2_tables_before_cleanup = localnode2_tasks_handler.get_tables(
        request_id=request_id, context_id=context_id
    )

    controller._append_context_id_for_cleanup(
        context_id=context_id,
        node_ids=[
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

    now = time.time()
    end = now
    while (
        globalnode_tables_after_cleanup
        and localnode1_tables_after_cleanup
        and localnode2_tables_after_cleanup
        and end - now < WAIT_CLEANUP_TIME_LIMIT
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
        end = time.time()
        await asyncio.sleep(2)

    await controller.stop_node_registry()
    await controller.stop_cleanup_loop()
    # give some time for node registry and cleanup background tasks to finish gracefully
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


# @pytest.mark.skip(reason="just skip it")
@pytest.mark.asyncio
async def test_cleanup_node_down_algorithm_execution(
    init_data_globalnode,
    load_data_localnode1,
    load_data_localnodetmp,
    globalnode_node_service,
    localnode1_node_service,
    localnodetmp_node_service,
):

    # get tmp localnode node_id from config file
    localnodetmp_node_id = get_localnodetmp_node_id()

    controller = Controller()

    # start node registry
    await controller.start_node_registry()

    # wait until node registry gets the nodes info
    while not controller.get_all_local_nodes():
        await asyncio.sleep(1)

    # get all participating node tasks handlers
    is_localnodetmp_up = False
    while not is_localnodetmp_up:
        nodes_tasks_handlers_dto = controller._get_nodes_tasks_handlers(
            data_model=algorithm_request_dto.inputdata.data_model,
            datasets=algorithm_request_dto.inputdata.datasets,
        )
        for (
            local_node_tasks_handler
        ) in nodes_tasks_handlers_dto.local_nodes_tasks_handlers:
            globalnode_tasks_handler = (
                nodes_tasks_handlers_dto.global_node_tasks_handler
            )
            localnode1_tasks_handler = (
                nodes_tasks_handlers_dto.local_nodes_tasks_handlers[0]
            )
            if local_node_tasks_handler.node_id == localnodetmp_node_id:
                localnodetmp_tasks_handler = local_node_tasks_handler
                is_localnodetmp_up = True

        await asyncio.sleep(1)

    globalnode_node_id = globalnode_tasks_handler.node_id
    localnode1_node_id = localnode1_tasks_handler.node_id

    request_id = get_a_uniqueid()
    context_id = get_a_uniqueid()
    algorithm_name = "logistic_regression"
    algo_execution_logger = ctrl_logger.get_request_logger(request_id=request_id)

    # start executing the algorithm but do not wait for it
    task = asyncio.create_task(
        controller._exec_algorithm_with_task_handlers(
            request_id=request_id,
            context_id=context_id,
            algorithm_name=algorithm_name,
            algorithm_request_dto=algorithm_request_dto,
            tasks_handlers=nodes_tasks_handlers_dto,
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

    # wait for get_tables tasks to return
    await asyncio.sleep(10)

    remove_localnodetmp_rabbitmq()

    # Start the cleanup loop
    cleanup_task = await controller.start_cleanup_loop()

    controller._append_context_id_for_cleanup(
        context_id=context_id,
        node_ids=[globalnode_node_id, localnode1_node_id, localnodetmp_node_id],
    )

    # restart tmplocalnode rabbitmq container
    _create_rabbitmq_container(RABBITMQ_LOCALNODETMP_NAME, RABBITMQ_LOCALNODETMP_PORT)
    start_localnodetmp_node_service()

    is_tmplocalnode_up = False
    while not is_tmplocalnode_up:
        nodes_tasks_handlers_dto = controller._get_nodes_tasks_handlers(
            data_model=algorithm_request_dto.inputdata.data_model,
            datasets=algorithm_request_dto.inputdata.datasets,
        )
        for (
            local_node_tasks_handler
        ) in nodes_tasks_handlers_dto.local_nodes_tasks_handlers:
            if local_node_tasks_handler.node_id == localnodetmp_node_id:
                localnodetmp_tasks_handler = local_node_tasks_handler
                is_tmplocalnode_up = True

        await asyncio.sleep(1)

    globalnode_tables_after_cleanup = globalnode_tasks_handler.get_tables(
        request_id=request_id, context_id=context_id
    )
    localnode1_tables_after_cleanup = localnode1_tasks_handler.get_tables(
        request_id=request_id, context_id=context_id
    )
    # again... find the localnodetmp potentially among other local nodes
    for local_node_tasks_handler in nodes_tasks_handlers_dto.local_nodes_tasks_handlers:
        if local_node_tasks_handler.node_id == get_localnodetmp_node_id():
            localnodetmp_tasks_handler = local_node_tasks_handler

    localnodetmp_tables_after_cleanup = localnodetmp_tasks_handler.get_tables(
        request_id=request_id, context_id=context_id
    )

    now = time.time()
    end = now
    while (
        globalnode_tables_after_cleanup
        and localnode1_tables_after_cleanup
        and localnodetmp_tables_after_cleanup
        and end - now < WAIT_CLEANUP_TIME_LIMIT
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
        end = time.time()
        await asyncio.sleep(2)

    await controller.stop_node_registry()
    await controller.stop_cleanup_loop()

    # give some time for node registry and cleanup background tasks to finish gracefully
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


algorithm_request_dto = AlgorithmRequestDTO(
    inputdata=AlgorithmInputDataDTO(
        data_model="dementia:0.1",
        datasets=["edsd"],
        filters={
            "condition": "AND",
            "rules": [
                {
                    "id": "dataset",
                    "type": "string",
                    "value": ["edsd"],
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


# @pytest.fixture(autouse=True, scope="session")
# def mock_controller_config():
#     import envtoml
#     from mipengine import AttrDict

#     this_mod_path = os.path.dirname(os.path.abspath(__file__))
#     TEST_ENV_CONFIG_FOLDER = path.join(this_mod_path, "testing_env_configs")

#     CONTROLLER_CONFIG_FILE = "test_controller_config.toml"
#     controller_config_file = path.join(TEST_ENV_CONFIG_FOLDER, CONTROLLER_CONFIG_FILE)

#     NODES_ADDRESSES_FILE = "test_localnodes_addresses.json"
#     nodes_addresses_file = path.join(TEST_ENV_CONFIG_FOLDER, NODES_ADDRESSES_FILE)

#     with open(controller_config_file) as fp:
#         controller_config = AttrDict(envtoml.load(fp))
#         with patch(
#             "mipengine.controller.config",
#             controller_config,
#         ):
#             yield

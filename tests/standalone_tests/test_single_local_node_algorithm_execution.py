import asyncio
from os import path
from unittest.mock import patch

import pytest
import toml

from mipengine import AttrDict
from mipengine.algorithm_result_DTOs import TabularDataResult
from mipengine.controller import controller_logger as ctrl_logger
from mipengine.controller.algorithm_execution_DTOs import AlgorithmExecutionDTO
from mipengine.controller.algorithm_execution_DTOs import NodesTasksHandlersDTO
from mipengine.controller.api.algorithm_request_dto import AlgorithmInputDataDTO
from mipengine.controller.api.algorithm_request_dto import AlgorithmRequestDTO
from mipengine.controller.controller import Controller
from mipengine.controller.controller import get_a_uniqueid
from mipengine.controller.data_model_registry import DataModelRegistry
from mipengine.node_tasks_DTOs import CommonDataElement
from mipengine.node_tasks_DTOs import CommonDataElements
from tests.standalone_tests.conftest import LOCALNODETMP_CONFIG_FILE
from tests.standalone_tests.conftest import TEST_ENV_CONFIG_FOLDER

WAIT_BACKGROUND_TASKS_TO_FINISH = 20


@pytest.fixture(scope="session")
def controller_config_mock():
    controller_config = AttrDict(
        {
            "log_level": "DEBUG",
            "framework_log_level": "INFO",
            "cdes_metadata_path": "./tests/demo_data",
            "deployment_type": "LOCAL",
            "node_landscape_aggregator_update_interval": 2,
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
                "celery_tasks_timeout": 30,
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


@pytest.mark.skip
@pytest.mark.slow
@pytest.mark.asyncio
async def test_single_local_node_algorithm_execution(
    globalnode_node_service,
    load_data_localnodetmp,
    localnodetmp_node_service,
):
    # local_node_id = "localnode1"
    localnodetmp_node_id = get_localnodetmp_node_id()

    controller = Controller()
    # start node landscape aggregator
    await controller.start_node_landscape_aggregator()

    # wait until node registry gets the nodes info
    while not controller.get_all_local_nodes() or not controller.get_global_node():
        await asyncio.sleep(1)

    # wait for localnodetmp to be available in node registry
    is_localnodetmp_up = False
    while not is_localnodetmp_up:
        nodes_tasks_handlers_dto = controller._get_nodes_tasks_handlers(
            data_model=algorithm_request_dto.inputdata.data_model,
            datasets=algorithm_request_dto.inputdata.datasets,
        )
        for (
            local_node_tasks_handler
        ) in nodes_tasks_handlers_dto.local_nodes_tasks_handlers:

            if local_node_tasks_handler.node_id == localnodetmp_node_id:
                globalnode_tasks_handler = (
                    nodes_tasks_handlers_dto.global_node_tasks_handler
                )
                localnodetmp_tasks_handler = local_node_tasks_handler
                # use only 1 localnode
                nodes_tasks_handlers_dto = NodesTasksHandlersDTO(
                    global_node_tasks_handler=globalnode_tasks_handler,
                    local_nodes_tasks_handlers=[localnodetmp_tasks_handler],
                )
                is_localnodetmp_up = True
        await asyncio.sleep(1)

    request_id = get_a_uniqueid()
    context_id = get_a_uniqueid()
    algorithm_name = "logistic_regression"
    algo_execution_logger = ctrl_logger.get_request_logger(request_id=request_id)

    result = await controller._exec_algorithm_with_task_handlers(
        request_id=request_id,
        context_id=context_id,
        algorithm_name=algorithm_name,
        algorithm_request_dto=algorithm_request_dto,
        tasks_handlers=nodes_tasks_handlers_dto,
        logger=algo_execution_logger,
    )

    await controller.stop_node_landscape_aggregator()
    # give some time for node registry background task to finish gracefully
    await asyncio.sleep(WAIT_BACKGROUND_TASKS_TO_FINISH)

    tbr = TabularDataResult.parse_raw(result)
    assert isinstance(tbr, TabularDataResult)


def get_localnodetmp_node_id():
    local_node_filepath = path.join(TEST_ENV_CONFIG_FOLDER, LOCALNODETMP_CONFIG_FILE)
    with open(local_node_filepath) as fp:
        tmp = toml.load(fp)
        node_id = tmp["identifier"]
    return node_id


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


# -------------------------------------------------------------------------------
@pytest.fixture(scope="function")
def mock_cdes():
    data_model_registry = DataModelRegistry()
    data_models = {
        "dementia:0.1": CommonDataElements(
            values={
                "lefthippocampus": CommonDataElement(
                    code="lefthippocampus",
                    label="Left Hippocampus",
                    sql_type="real",
                    isCategorical=False,
                ),
                "righthippocampus": CommonDataElement(
                    code="righthippocampus",
                    label="Right Hippocampus",
                    sql_type="real",
                    isCategorical=False,
                ),
                "rightppplanumpolare": CommonDataElement(
                    code="rightppplanumpolare",
                    label="Right planum polare",
                    sql_type="real",
                    isCategorical=False,
                ),
                "leftamygdala": CommonDataElement(
                    code="leftamygdala",
                    label="Left Amygdala",
                    sql_type="real",
                    isCategorical=False,
                ),
                "rightamygdala": CommonDataElement(
                    code="rightamygdala",
                    label="Right Amygdala",
                    sql_type="real",
                    isCategorical=False,
                ),
                "alzheimerbroadcategory": CommonDataElement(
                    code="alzheimerbroadcategory",
                    label="There will be two broad categories taken into account. Alzheimer s disease (AD) in which the diagnostic is 100% certain and <Other> comprising the rest of Alzheimer s related categories. The <Other> category refers to Alzheime s related diagnosis which origin can be traced to other pathology eg. vascular. In this category MCI diagnosis can also be found. In summary  all Alzheimer s related diagnosis that are not pure.",
                    sql_type="text",
                    isCategorical=True,
                    enumerations={
                        "AD": "Alzheimer’s disease",
                        "CN": "Cognitively Normal",
                    },
                ),
            }
        )
    }
    data_model_registry.set_data_models(data_models)
    with patch(
        "mipengine.controller.algorithm_executor.data_model_registry",
        data_model_registry,
    ):
        yield


@pytest.fixture(scope="function")
def mock_ctrl_config():
    ctrl_config = AttrDict(
        {
            "smpc": {
                "enabled": False,
                "optional": False,
            }
        }
    )

    with patch(
        "mipengine.controller.algorithm_executor.ctrl_config",
        ctrl_config,
    ):
        yield


@pytest.fixture(scope="function")
def mock_algorithms_modules():
    import mipengine

    mipengine.ALGORITHM_FOLDERS = "./mipengine/algorithms,./tests/algorithms"
    algorithm_modules = mipengine.import_algorithm_modules()

    with patch(
        "mipengine.controller.algorithm_executor.algorithm_modules",
        algorithm_modules,
    ):
        yield

    mipengine.ALGORITHM_FOLDERS = "./mipengine/algorithms"


def get_parametrization_list_success_cases():
    parametrization_list = []

    # ~~~~~~~~~~success case 1~~~~~~~~~~
    algo_execution_dto = AlgorithmExecutionDTO(
        request_id="123",
        context_id="123",
        algorithm_name="logistic_regression",
        algorithm_request_dto=AlgorithmRequestDTO(
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
        ),
    )
    parametrization_list.append(algo_execution_dto)
    # END ~~~~~~~~~~success case 1~~~~~~~~~~

    # ~~~~~~~~~~success case 2~~~~~~~~~~
    algo_execution_dto = AlgorithmExecutionDTO(
        request_id="1234",
        context_id="1234",
        algorithm_name="smpc_standard_deviation",
        algorithm_request_dto=AlgorithmRequestDTO(
            request_id="1234",
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
                                ]
                            ],
                        },
                    ],
                    "valid": True,
                },
                x=[
                    "lefthippocampus",
                ],
            ),
            parameters={"classes": ["AD", "CN"]},
            flags={"smpc": False},
        ),
    )
    parametrization_list.append(algo_execution_dto)
    # END ~~~~~~~~~~success case 2~~~~~~~~~~
    return parametrization_list


# @pytest.mark.parametrize(
#     "algo_execution_dto",
#     get_parametrization_list_success_cases(),
# )
# def test_single_local_node_algorithm_execution(
#     mock_cdes, mock_ctrl_config, mock_algorithms_modules, algo_execution_dto
# ):
#     local_node_id = "localnode1"

#     node_config = get_node_config_by_id(local_node_id)
#     queue_addr = str(node_config.rabbitmq.ip) + ":" + str(node_config.rabbitmq.port)
#     db_addr = str(node_config.monetdb.ip) + ":" + str(node_config.monetdb.port)
#     local_node_task_handler = NodeTasksHandlerCelery(
#         node_id=local_node_id,
#         node_queue_addr=queue_addr,
#         node_db_addr=db_addr,
#         tasks_timeout=45,
#     )

#     single_node_task_handler = NodesTasksHandlersDTO(
#         global_node_tasks_handler=local_node_task_handler,
#         local_nodes_tasks_handlers=[local_node_task_handler],
#     )
#     algo_executor = AlgorithmExecutor(
#         algorithm_execution_dto=algo_execution_dto,
#         nodes_tasks_handlers_dto=single_node_task_handler,
#     )
#     result = algo_executor.run()

#     assert isinstance(result, TabularDataResult)

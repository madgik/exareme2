from unittest.mock import patch

import pytest

from mipengine import AttrDict
from mipengine.algorithm_result_DTOs import TabularDataResult
from mipengine.controller.algorithm_execution_DTOs import AlgorithmExecutionDTO
from mipengine.controller.algorithm_execution_DTOs import NodesTasksHandlersDTO
from mipengine.controller.algorithm_execution_tasks_handler import (
    NodeAlgorithmTasksHandler,
)
from mipengine.controller.algorithm_executor import AlgorithmExecutor
from mipengine.node_tasks_DTOs import CommonDataElement
from tests.standalone_tests.conftest import MONETDB_LOCALNODE1_PORT
from tests.standalone_tests.conftest import RABBITMQ_LOCALNODE1_PORT


@pytest.fixture(scope="function")
def mock_common_data_elements():
    common_data_elements = {
        "lefthippocampus": CommonDataElement(
            code="lefthippocampus",
            label="Left Hippocampus",
            sql_type="real",
            is_categorical=False,
        ),
        "righthippocampus": CommonDataElement(
            code="righthippocampus",
            label="Right Hippocampus",
            sql_type="real",
            is_categorical=False,
        ),
        "rightppplanumpolare": CommonDataElement(
            code="rightppplanumpolare",
            label="Right planum polare",
            sql_type="real",
            is_categorical=False,
        ),
        "leftamygdala": CommonDataElement(
            code="leftamygdala",
            label="Left Amygdala",
            sql_type="real",
            is_categorical=False,
        ),
        "rightamygdala": CommonDataElement(
            code="rightamygdala",
            label="Right Amygdala",
            sql_type="real",
            is_categorical=False,
        ),
        "alzheimerbroadcategory": CommonDataElement(
            code="alzheimerbroadcategory",
            label="There will be two broad categories taken into account. Alzheimer s disease (AD) in which the diagnostic is 100% certain and <Other> comprising the rest of Alzheimer s related categories. The <Other> category refers to Alzheime s related diagnosis which origin can be traced to other pathology eg. vascular. In this category MCI diagnosis can also be found. In summary  all Alzheimer s related diagnosis that are not pure.",
            sql_type="text",
            is_categorical=True,
            enumerations={
                "AD": "Alzheimer’s disease",
                "CN": "Cognitively Normal",
            },
        ),
    }
    yield common_data_elements


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
        data_model="dementia:0.1",
        datasets_per_local_node={
            "localnode1": [
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
            ]
        },
        x_vars=[
            "lefthippocampus",
            "righthippocampus",
            "rightppplanumpolare",
            "leftamygdala",
            "rightamygdala",
        ],
        y_vars=["alzheimerbroadcategory"],
        var_filters={
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
        },
        algo_parameters={"classes": ["AD", "CN"]},
    )
    parametrization_list.append(algo_execution_dto)
    # END ~~~~~~~~~~success case 1~~~~~~~~~~

    # ~~~~~~~~~~success case 2~~~~~~~~~~
    algo_execution_dto = AlgorithmExecutionDTO(
        request_id="1234",
        context_id="1234",
        algorithm_name="smpc_standard_deviation",
        data_model="dementia:0.1",
        datasets_per_local_node={
            "localnode1": [
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
            ]
        },
        x_vars=[
            "lefthippocampus",
        ],
        var_filters={
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
                        ]
                    ],
                },
            ],
        },
        algo_parameters={"classes": ["AD", "CN"]},
        algo_flags={"smpc": False},
    )
    parametrization_list.append(algo_execution_dto)
    # END ~~~~~~~~~~success case 2~~~~~~~~~~
    return parametrization_list


@pytest.mark.parametrize(
    "algo_execution_dto",
    get_parametrization_list_success_cases(),
)
def test_single_local_node_algorithm_execution(
    mock_common_data_elements,
    mock_ctrl_config,
    mock_algorithms_modules,
    algo_execution_dto,
    globalnode_node_service,
    localnode1_node_service,
    load_data_localnode1,
):
    local_node_id = "localnode1"
    local_node_ip = "172.17.0.1"
    local_node_monetdb_port = MONETDB_LOCALNODE1_PORT
    local_node_rabbitmq_port = RABBITMQ_LOCALNODE1_PORT
    queue_addr = local_node_ip + ":" + str(local_node_rabbitmq_port)
    db_addr = local_node_ip + ":" + str(local_node_monetdb_port)
    local_node_task_handler = NodeAlgorithmTasksHandler(
        node_id=local_node_id,
        node_queue_addr=queue_addr,
        node_db_addr=db_addr,
        tasks_timeout=45,
    )

    single_node_task_handler = NodesTasksHandlersDTO(
        global_node_tasks_handler=local_node_task_handler,
        local_nodes_tasks_handlers=[local_node_task_handler],
    )
    algo_executor = AlgorithmExecutor(
        algorithm_execution_dto=algo_execution_dto,
        nodes_tasks_handlers_dto=single_node_task_handler,
        common_data_elements=mock_common_data_elements,
    )
    result = algo_executor.run()

    assert isinstance(result, TabularDataResult)

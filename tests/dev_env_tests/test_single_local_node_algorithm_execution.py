import pytest

from tests.dev_env_tests.nodes_communication import get_celery_app
from tests.dev_env_tests.nodes_communication import get_node_config_by_id

from mipengine.controller.node_tasks_handler_celery import NodeTasksHandlerCelery
from mipengine.controller.algorithm_execution_DTOs import AlgorithmExecutionDTO
from mipengine.controller.algorithm_execution_DTOs import NodesTasksHandlersDTO
from mipengine.controller.api.algorithm_request_dto import AlgorithmRequestDTO
from mipengine.controller.api.algorithm_request_dto import AlgorithmInputDataDTO

from mipengine.algorithm_flow_DTOs import TabularDataResult

from mipengine.table_data_DTOs import ColumnDataFloat
from mipengine.table_data_DTOs import ColumnDataStr
from mipengine import DType

from mipengine.controller.algorithm_executor import AlgorithmExecutor


algo_execution_dto = AlgorithmExecutionDTO(
    context_id="123",
    algorithm_name="logistic_regression",
    algorithm_request_dto=AlgorithmRequestDTO(
        inputdata=AlgorithmInputDataDTO(
            pathology="dementia",
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


def test_single_local_node_algorithm_execution():

    local_node_id = "localnode1"
    global_node_id = "globalnode"

    # GLOBALNODE
    node_config = get_node_config_by_id(global_node_id)
    queue_addr = str(node_config.rabbitmq.ip) + ":" + str(node_config.rabbitmq.port)
    db_addr = str(node_config.monetdb.ip) + ":" + str(node_config.monetdb.port)
    global_node_task_handler = NodeTasksHandlerCelery(
        node_id=global_node_id,
        node_queue_addr=queue_addr,
        node_db_addr=db_addr,
        tasks_timeout=45,
    )

    # LOCALNODE
    node_config = get_node_config_by_id(local_node_id)
    queue_addr = str(node_config.rabbitmq.ip) + ":" + str(node_config.rabbitmq.port)
    db_addr = str(node_config.monetdb.ip) + ":" + str(node_config.monetdb.port)
    local_node_task_handler = NodeTasksHandlerCelery(
        node_id=local_node_id,
        node_queue_addr=queue_addr,
        node_db_addr=db_addr,
        tasks_timeout=45,
    )

    single_node_task_handler = NodesTasksHandlersDTO(
        global_node_tasks_handler=global_node_task_handler,
        local_nodes_tasks_handlers=[local_node_task_handler],
    )
    algo_executor = AlgorithmExecutor(
        algorithm_execution_dto=algo_execution_dto,
        nodes_tasks_handlers_dto=single_node_task_handler,
    )
    result = algo_executor.run()

    assert isinstance(result, TabularDataResult)

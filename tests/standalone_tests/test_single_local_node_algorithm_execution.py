from os import path

import pytest
from pydantic import BaseModel

from mipengine import AttrDict
from mipengine import algorithm_classes
from mipengine.algorithms.algorithm import AlgorithmDTO
from mipengine.algorithms.algorithm import Variables
from mipengine.controller.algorithm_executor import (
    InitializationParams as AlgorithmExecutorInitParams,
)
from mipengine.controller.api.algorithm_request_dto import AlgorithmInputDataDTO
from mipengine.controller.api.algorithm_request_dto import AlgorithmRequestDTO
from mipengine.controller.controller import CommandIdGenerator
from mipengine.controller.controller import Controller
from mipengine.controller.controller import InitializationParams as ControllerInitParams
from mipengine.controller.controller import Nodes
from mipengine.controller.controller import _create_algorithm_executor
from mipengine.controller.controller import _filter_insufficient_data_nodes
from mipengine.controller.controller import sanitize_request_variable
from mipengine.controller.node_landscape_aggregator import (
    InitializationParams as NodeLandscapeAggregatorInitParams,
)
from mipengine.controller.node_landscape_aggregator import NodeLandscapeAggregator
from mipengine.controller.uid_generator import UIDGenerator
from tests.standalone_tests.conftest import CONTROLLER_LOCALNODE1_ADDRESSES_FILE
from tests.standalone_tests.conftest import TEST_ENV_CONFIG_FOLDER


@pytest.fixture
def command_id_generator():
    return CommandIdGenerator()


@pytest.fixture
def controller_config():
    controller_config = {
        "log_level": "DEBUG",
        "framework_log_level": "INFO",
        "deployment_type": "LOCAL",
        "node_landscape_aggregator_update_interval": 30,
        # "cleanup": {
        #     "contextids_cleanup_folder": "/tmp",
        #     "nodes_cleanup_interval": 9999,  # don't cleanup
        #     "contextid_release_timelimit": 3600,  # 1hour
        # },
        "localnodes": {
            "config_file": path.join(
                TEST_ENV_CONFIG_FOLDER, CONTROLLER_LOCALNODE1_ADDRESSES_FILE
            ),
            "dns": "",
            "port": "",
        },
        "rabbitmq": {
            "user": "user",
            "password": "password",
            "vhost": "user_vhost",
            "celery_tasks_timeout": 5,
            "celery_run_udf_task_timeout": 10,
            "celery_tasks_max_retries": 3,
            "celery_tasks_interval_start": 0,
            "celery_tasks_interval_step": 0.2,
            "celery_tasks_interval_max": 0.5,
            "celery_cleanup_task_timeout": 2,
        },
        "smpc": {"enabled": False, "optional": False},
    }
    return controller_config


@pytest.fixture
def node_landscape_aggregator(
    controller_config, localnode1_node_service, load_data_localnode1
):
    controller_config = AttrDict(controller_config)
    node_landscape_aggregator_init_params = NodeLandscapeAggregatorInitParams(
        node_landscape_aggregator_update_interval=controller_config.node_landscape_aggregator_update_interval,
        celery_tasks_timeout=controller_config.rabbitmq.celery_tasks_timeout,
        celery_run_udf_task_timeout=controller_config.rabbitmq.celery_run_udf_task_timeout,
        deployment_type=controller_config.deployment_type,
        localnodes=controller_config.localnodes,
    )
    node_landscape_aggregator = NodeLandscapeAggregator(
        node_landscape_aggregator_init_params
    )
    node_landscape_aggregator._update()
    node_landscape_aggregator.start()
    return node_landscape_aggregator


@pytest.fixture
def datasets():
    return [
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


@pytest.fixture
def algorithm_request_case_1(datasets):
    algorithm_name = "pca"
    algorithm_request_dto = AlgorithmRequestDTO(
        request_id=UIDGenerator().get_a_uid(),
        inputdata=AlgorithmInputDataDTO(
            data_model="dementia:0.1",
            datasets=datasets,
            filters={
                "condition": "AND",
                "rules": [
                    {
                        "id": "dataset",
                        "type": "string",
                        "value": datasets,
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
                            ]
                        ],
                    },
                ],
            },
            x=[],
            y=[
                "lefthippocampus",
                "righthippocampus",
                "rightppplanumpolare",
                "leftamygdala",
                "rightamygdala",
            ],
        ),
        parameters={},
    )

    return (algorithm_name, algorithm_request_dto)


@pytest.fixture()
def algorithm_request_case_2(datasets):
    algorithm_name = "logistic_regression"
    algo_request_dto = AlgorithmRequestDTO(
        request_id=UIDGenerator().get_a_uid(),
        inputdata=AlgorithmInputDataDTO(
            data_model="dementia:0.1",
            datasets=datasets,
            filters={
                "condition": "AND",
                "rules": [
                    {
                        "id": "dataset",
                        "type": "string",
                        "value": datasets,
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
            x=[
                "lefthippocampus",
                "righthippocampus",
                "rightppplanumpolare",
                "leftamygdala",
                "rightamygdala",
            ],
            y=["alzheimerbroadcategory"],
        ),
        parameters={"positive_class": "AD", "positive_class": "CN"},
    )
    return (algorithm_name, algo_request_dto)


@pytest.fixture
def metadata_case_1(node_landscape_aggregator, algorithm_request_case_1):
    algorithm_request_dto = algorithm_request_case_1[1]

    variable_names = (algorithm_request_dto.inputdata.x or []) + (
        algorithm_request_dto.inputdata.y or []
    )
    return node_landscape_aggregator.get_metadata(
        data_model=algorithm_request_dto.inputdata.data_model,
        variable_names=variable_names,
    )


@pytest.fixture(scope="function")
def metadata_case_2(node_landscape_aggregator, algorithm_request_case_2):
    algorithm_request_dto = algorithm_request_case_2[1]

    variable_names = (algorithm_request_dto.inputdata.x or []) + (
        algorithm_request_dto.inputdata.y or []
    )
    return node_landscape_aggregator.get_metadata(
        data_model=algorithm_request_dto.inputdata.data_model,
        variable_names=variable_names,
    )


@pytest.fixture
def algorithm_case_1(algorithm_request_case_1, metadata_case_1):
    algorithm_name = algorithm_request_case_1[0]
    algorithm_request_dto = algorithm_request_case_1[1]

    input_data = algorithm_request_dto.inputdata
    algorithm_parameters = algorithm_request_dto.parameters

    algorithm_dto = AlgorithmDTO(
        algorithm_name=algorithm_name,
        data_model=input_data.data_model,
        variables=Variables(
            x=sanitize_request_variable(input_data.x),
            y=sanitize_request_variable(input_data.y),
        ),
        var_filters=input_data.filters,
        algorithm_parameters=algorithm_parameters,
        metadata=metadata_case_1,
    )
    return algorithm_classes[algorithm_name](algorithm_dto=algorithm_dto)


@pytest.fixture(scope="function")
def algorithm_case_2(algorithm_request_case_2, metadata_case_2):
    algorithm_name = algorithm_request_case_2[0]
    algorithm_request_dto = algorithm_request_case_2[1]

    input_data = algorithm_request_dto.inputdata
    algorithm_parameters = algorithm_request_dto.parameters

    algorithm_dto = AlgorithmDTO(
        algorithm_name=algorithm_name,
        data_model=input_data.data_model,
        variables=Variables(
            x=sanitize_request_variable(input_data.x),
            y=sanitize_request_variable(input_data.y),
        ),
        var_filters=input_data.filters,
        algorithm_parameters=algorithm_parameters,
        metadata=metadata_case_2,
    )
    return algorithm_classes[algorithm_name](algorithm_dto=algorithm_dto)


@pytest.fixture
def context_id():
    return UIDGenerator().get_a_uid()


@pytest.fixture
def nodes_case_1(controller, context_id, algorithm_request_case_1):
    algorithm_request_dto = algorithm_request_case_1[1]

    return controller._create_nodes(
        request_id=algorithm_request_dto.request_id,
        context_id=context_id,
        data_model=algorithm_request_dto.inputdata.data_model,
        datasets=algorithm_request_dto.inputdata.datasets,
    )


@pytest.fixture(scope="function")
def nodes_case_2(controller, context_id, algorithm_request_case_2):
    algorithm_request_dto = algorithm_request_case_2[1]

    return controller._create_nodes(
        request_id=algorithm_request_dto.request_id,
        context_id=context_id,
        data_model=algorithm_request_dto.inputdata.data_model,
        datasets=algorithm_request_dto.inputdata.datasets,
    )


@pytest.fixture
def controller(controller_config, node_landscape_aggregator):
    controller_config = AttrDict(controller_config)

    controller_init_params = ControllerInitParams(
        smpc_enabled=False,
        smpc_optional=False,
        celery_tasks_timeout=controller_config.rabbitmq.celery_tasks_timeout,
        celery_run_udf_task_timeout=controller_config.rabbitmq.celery_run_udf_task_timeout,
    )
    controller = Controller(
        initialization_params=controller_init_params,
        cleaner=None,
        node_landscape_aggregator=node_landscape_aggregator,
    )
    return controller


@pytest.fixture
def data_model_views_and_nodes_case_1(
    datasets,
    algorithm_request_case_1,
    nodes_case_1,
    algorithm_case_1,
    command_id_generator,
    controller,
):
    algorithm_request_dto = algorithm_request_case_1[1]
    nodes = nodes_case_1

    data_model_views = controller._create_data_model_views(
        local_nodes=nodes.local_nodes,
        datasets=datasets,
        data_model=algorithm_request_dto.inputdata.data_model,
        variable_groups=algorithm_case_1.get_variable_groups(),
        var_filters=algorithm_request_dto.inputdata.filters,
        dropna=algorithm_case_1.get_dropna(),
        check_min_rows=algorithm_case_1.get_check_min_rows(),
        command_id=command_id_generator.get_next_command_id(),
    )
    local_nodes_filtered = _filter_insufficient_data_nodes(
        nodes.local_nodes, data_model_views
    )
    nodes = Nodes(global_node=nodes.global_node, local_nodes=local_nodes_filtered)
    if not nodes.local_nodes:
        raise RequestConstraintsError(algorithm_request_dto)
    return (data_model_views, nodes)


@pytest.fixture(scope="function")
def data_model_views_and_nodes_case_2(
    datasets,
    algorithm_request_case_2,
    nodes_case_2,
    algorithm_case_2,
    command_id_generator,
    controller,
):
    algorithm_request_dto = algorithm_request_case_2[1]
    nodes = nodes_case_2

    data_model_views = controller._create_data_model_views(
        local_nodes=nodes.local_nodes,
        datasets=datasets,
        data_model=algorithm_request_dto.inputdata.data_model,
        variable_groups=algorithm_case_2.get_variable_groups(),
        var_filters=algorithm_request_dto.inputdata.filters,
        dropna=algorithm_case_2.get_dropna(),
        check_min_rows=algorithm_case_2.get_check_min_rows(),
        command_id=command_id_generator.get_next_command_id(),
    )
    local_nodes_filtered = _filter_insufficient_data_nodes(
        nodes.local_nodes, data_model_views
    )
    nodes = Nodes(global_node=nodes.global_node, local_nodes=local_nodes_filtered)
    if not nodes.local_nodes:
        raise RequestConstraintsError(algorithm_request_dto)
    return (data_model_views, nodes)


@pytest.fixture
def algorithm_executor_case_1(
    algorithm_request_case_1,
    context_id,
    data_model_views_and_nodes_case_1,
    command_id_generator,
):
    algorithm_request_dto = algorithm_request_case_1[1]

    algorithm_executor_init_params = AlgorithmExecutorInitParams(
        smpc_enabled=False,
        smpc_optional=False,
        request_id=algorithm_request_dto.request_id,
        context_id=context_id,
        algo_flags=algorithm_request_dto.flags,
        data_model_views=data_model_views_and_nodes_case_1[0],
    )
    return _create_algorithm_executor(
        algorithm_executor_init_params=algorithm_executor_init_params,
        command_id_generator=command_id_generator,
        nodes=data_model_views_and_nodes_case_1[1],
    )


@pytest.fixture(scope="function")
def algorithm_executor_case_2(
    algorithm_request_case_2,
    context_id,
    data_model_views_and_nodes_case_2,
    command_id_generator,
):
    algorithm_request_dto = algorithm_request_case_2[1]

    algorithm_executor_init_params = AlgorithmExecutorInitParams(
        smpc_enabled=False,
        smpc_optional=False,
        request_id=algorithm_request_dto.request_id,
        context_id=context_id,
        algo_flags=algorithm_request_dto.flags,
        data_model_views=data_model_views_and_nodes_case_2[0],
    )
    return _create_algorithm_executor(
        algorithm_executor_init_params=algorithm_executor_init_params,
        command_id_generator=command_id_generator,
        nodes=data_model_views_and_nodes_case_2[1],
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "algorithm,algorithm_executor",
    [
        ("algorithm_case_1", "algorithm_executor_case_1"),
        ("algorithm_case_2", "algorithm_executor_case_2"),
    ],
)
@pytest.mark.asyncio
async def test_single_local_node_algorithm_execution(
    algorithm,
    algorithm_executor,
    controller,
    request,
):
    algorithm = request.getfixturevalue(algorithm)
    algorithm_executor = request.getfixturevalue(algorithm_executor)

    try:
        algorithm_result = await controller._algorithm_run_in_event_loop(
            algorithm=algorithm, algorithm_executor=algorithm_executor
        )
    except Exception as exc:
        pytest.fail(f"Execution of the algorithm failed with {exc=}")

    assert isinstance(algorithm_result, BaseModel)

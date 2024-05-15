from os import path

import pytest

from exareme2 import AttrDict
from exareme2 import algorithm_classes
from exareme2 import algorithm_data_loaders
from exareme2.algorithms.exareme2.algorithm import (
    InitializationParams as AlgorithmInitParams,
)
from exareme2.algorithms.exareme2.algorithm import Variables
from exareme2.algorithms.specifications import AlgorithmType
from exareme2.controller import logger as ctrl_logger
from exareme2.controller.services.api.algorithm_request_dtos import (
    AlgorithmInputDataDTO,
)
from exareme2.controller.services.api.algorithm_request_dtos import AlgorithmRequestDTO
from exareme2.controller.services.exareme2.controller import Controller
from exareme2.controller.services.exareme2.controller import DataModelViewsCreator
from exareme2.controller.services.exareme2.controller import WorkersFederation
from exareme2.controller.services.exareme2.controller import (
    _algorithm_run_in_event_loop,
)
from exareme2.controller.services.exareme2.controller import (
    _create_algorithm_execution_engine,
)
from exareme2.controller.services.exareme2.controller import sanitize_request_variable
from exareme2.controller.services.exareme2.execution_engine import CommandIdGenerator
from exareme2.controller.services.exareme2.execution_engine import (
    InitializationParams as EngineInitParams,
)
from exareme2.controller.services.exareme2.execution_engine import SMPCParams
from exareme2.controller.services.exareme2.execution_engine import Workers
from exareme2.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    WorkerLandscapeAggregator,
)
from exareme2.controller.uid_generator import UIDGenerator
from tests.standalone_tests.conftest import CONTROLLER_LOCALWORKER1_ADDRESSES_FILE
from tests.standalone_tests.conftest import TEST_ENV_CONFIG_FOLDER


@pytest.fixture(autouse=True, scope="session")
def init_background_controller_logger():
    ctrl_logger.set_background_service_logger("DEBUG")


@pytest.fixture(scope="function")
def command_id_generator():
    return CommandIdGenerator()


@pytest.fixture
def controller_config():
    controller_config = {
        "log_level": "DEBUG",
        "framework_log_level": "INFO",
        "deployment_type": "LOCAL",
        "worker_landscape_aggregator_update_interval": 30,
        "flower_execution_timeout": 30,
        "localworkers": {
            "config_file": path.join(
                TEST_ENV_CONFIG_FOLDER, CONTROLLER_LOCALWORKER1_ADDRESSES_FILE
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
        "smpc": {"enabled": False, "optional": False, "dp": {"enabled": False}},
    }
    return controller_config


@pytest.fixture(scope="function")
def worker_landscape_aggregator(
    controller_config, localworker1_worker_service, load_data_localworker1
):
    controller_config = AttrDict(controller_config)

    worker_landscape_aggregator = WorkerLandscapeAggregator(
        logger=ctrl_logger.get_background_service_logger(),
        update_interval=controller_config.worker_landscape_aggregator_update_interval,
        tasks_timeout=controller_config.rabbitmq.celery_tasks_timeout,
        run_udf_task_timeout=controller_config.rabbitmq.celery_run_udf_task_timeout,
        deployment_type=controller_config.deployment_type,
        localworkers=controller_config.localworkers,
    )
    worker_landscape_aggregator.update()
    worker_landscape_aggregator.start()

    return worker_landscape_aggregator


@pytest.fixture
def datasets():
    return [
        "edsd0",
        "edsd1",
        "edsd2",
        "edsd3",
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
        type=AlgorithmType.EXAREME2,
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
        parameters={"positive_class": "AD"},
        type=AlgorithmType.EXAREME2,
    )
    return (algorithm_name, algo_request_dto)


@pytest.fixture(scope="function")
def metadata_case_1(worker_landscape_aggregator, algorithm_request_case_1):
    algorithm_request_dto = algorithm_request_case_1[1]

    variable_names = (algorithm_request_dto.inputdata.x or []) + (
        algorithm_request_dto.inputdata.y or []
    )
    return worker_landscape_aggregator.get_metadata(
        data_model=algorithm_request_dto.inputdata.data_model,
        variable_names=variable_names,
    )


@pytest.fixture(scope="function")
def metadata_case_2(worker_landscape_aggregator, algorithm_request_case_2):
    algorithm_request_dto = algorithm_request_case_2[1]

    variable_names = (algorithm_request_dto.inputdata.x or []) + (
        algorithm_request_dto.inputdata.y or []
    )
    return worker_landscape_aggregator.get_metadata(
        data_model=algorithm_request_dto.inputdata.data_model,
        variable_names=variable_names,
    )


@pytest.fixture(scope="function")
def algorithm_data_loader_case_1(algorithm_request_case_1):
    algorithm_name = algorithm_request_case_1[0]
    algorithm_request_dto = algorithm_request_case_1[1]
    algorithm_data_loader = algorithm_data_loaders[algorithm_name](
        variables=Variables(
            x=sanitize_request_variable(algorithm_request_dto.inputdata.x),
            y=sanitize_request_variable(algorithm_request_dto.inputdata.y),
        ),
    )
    return algorithm_data_loader


@pytest.fixture(scope="function")
def algorithm_data_loader_case_2(algorithm_request_case_2):
    algorithm_name = algorithm_request_case_2[0]
    algorithm_request_dto = algorithm_request_case_2[1]
    algorithm_data_loader = algorithm_data_loaders[algorithm_name](
        variables=Variables(
            x=sanitize_request_variable(algorithm_request_dto.inputdata.x),
            y=sanitize_request_variable(algorithm_request_dto.inputdata.y),
        ),
    )
    return algorithm_data_loader


@pytest.fixture(scope="function")
def algorithm_case_1(
    algorithm_request_case_1, algorithm_data_loader_case_1, engine_case_1
):
    algorithm_name = algorithm_request_case_1[0]
    algorithm_request_dto = algorithm_request_case_1[1]

    input_data = algorithm_request_dto.inputdata
    algorithm_parameters = algorithm_request_dto.parameters

    init_params = AlgorithmInitParams(
        algorithm_name=algorithm_name,
        data_model=input_data.data_model,
        variables=Variables(
            x=sanitize_request_variable(input_data.x),
            y=sanitize_request_variable(input_data.y),
        ),
        var_filters=input_data.filters,
        algorithm_parameters=algorithm_parameters,
        datasets=algorithm_request_dto.inputdata.datasets,
    )
    return algorithm_classes[algorithm_name](
        initialization_params=init_params,
        data_loader=algorithm_data_loader_case_1,
        engine=engine_case_1,
    )


@pytest.fixture(scope="function")
def algorithm_case_2(
    algorithm_request_case_2, algorithm_data_loader_case_2, engine_case_2
):
    algorithm_name = algorithm_request_case_2[0]
    algorithm_request_dto = algorithm_request_case_2[1]

    input_data = algorithm_request_dto.inputdata
    algorithm_parameters = algorithm_request_dto.parameters

    init_params = AlgorithmInitParams(
        algorithm_name=algorithm_name,
        data_model=input_data.data_model,
        variables=Variables(
            x=sanitize_request_variable(input_data.x),
            y=sanitize_request_variable(input_data.y),
        ),
        var_filters=input_data.filters,
        algorithm_parameters=algorithm_parameters,
        datasets=algorithm_request_dto.inputdata.datasets,
    )
    return algorithm_classes[algorithm_name](
        initialization_params=init_params,
        data_loader=algorithm_data_loader_case_2,
        engine=engine_case_2,
    )


@pytest.fixture(scope="function")
def context_id():
    return UIDGenerator().get_a_uid()


@pytest.fixture(scope="function")
def workers_case_1(
    controller,
    context_id,
    algorithm_request_case_1,
    worker_landscape_aggregator,
    controller_config,
):
    algorithm_request_dto = algorithm_request_case_1[1]

    controller_config = AttrDict(controller_config)

    workers_federation = WorkersFederation(
        request_id=algorithm_request_dto.request_id,
        context_id=context_id,
        data_model=algorithm_request_dto.inputdata.data_model,
        datasets=algorithm_request_dto.inputdata.datasets,
        var_filters=algorithm_request_dto.inputdata.filters,
        worker_landscape_aggregator=worker_landscape_aggregator,
        celery_tasks_timeout=controller_config.rabbitmq.celery_tasks_timeout,
        celery_run_udf_task_timeout=controller_config.rabbitmq.celery_run_udf_task_timeout,
        command_id_generator=command_id_generator,
        logger=ctrl_logger.get_request_logger(
            request_id=algorithm_request_dto.request_id
        ),
    )
    return workers_federation.workers


@pytest.fixture(scope="function")
def workers_case_2(
    controller,
    context_id,
    algorithm_request_case_2,
    worker_landscape_aggregator,
    controller_config,
):
    algorithm_request_dto = algorithm_request_case_2[1]

    controller_config = AttrDict(controller_config)

    workers_federation = WorkersFederation(
        request_id=algorithm_request_dto.request_id,
        context_id=context_id,
        data_model=algorithm_request_dto.inputdata.data_model,
        datasets=algorithm_request_dto.inputdata.datasets,
        var_filters=algorithm_request_dto.inputdata.filters,
        worker_landscape_aggregator=worker_landscape_aggregator,
        celery_tasks_timeout=controller_config.rabbitmq.celery_tasks_timeout,
        celery_run_udf_task_timeout=controller_config.rabbitmq.celery_run_udf_task_timeout,
        command_id_generator=command_id_generator,
        logger=ctrl_logger.get_request_logger(
            request_id=algorithm_request_dto.request_id
        ),
    )
    return workers_federation.workers


@pytest.fixture(scope="function")
def controller(controller_config, worker_landscape_aggregator):
    controller_config = AttrDict(controller_config)

    controller = Controller(
        worker_landscape_aggregator=worker_landscape_aggregator,
        cleaner=None,
        logger=ctrl_logger.get_background_service_logger(),
        tasks_timeout=controller_config.rabbitmq.celery_tasks_timeout,
        run_udf_task_timeout=controller_config.rabbitmq.celery_run_udf_task_timeout,
        smpc_params=SMPCParams(smpc_enabled=False, smpc_optional=False),
    )

    return controller


@pytest.fixture(scope="function")
def data_model_views_and_workers_case_1(
    datasets,
    algorithm_request_case_1,
    workers_case_1,
    algorithm_data_loader_case_1,
    command_id_generator,
    controller,
):
    algorithm_request_dto = algorithm_request_case_1[1]
    workers = workers_case_1

    data_model_views_creator = DataModelViewsCreator(
        local_workers=workers.local_workers,
        variable_groups=algorithm_data_loader_case_1.get_variable_groups(),
        var_filters=algorithm_request_dto.inputdata.filters,
        dropna=algorithm_data_loader_case_1.get_dropna(),
        check_min_rows=algorithm_data_loader_case_1.get_check_min_rows(),
        command_id=command_id_generator.get_next_command_id(),
    )
    data_model_views_creator.create_data_model_views()
    data_model_views = data_model_views_creator.data_model_views

    local_workers_filtered = (
        data_model_views_creator.data_model_views.get_list_of_workers()
    )
    if not local_workers_filtered:
        pytest.fail(
            f"None of the workers contains data to execute the request: {algorithm_request_dto=}"
        )

    workers = Workers(
        global_worker=workers.global_worker, local_workers=local_workers_filtered
    )
    return (data_model_views, workers)


@pytest.fixture(scope="function")
def data_model_views_and_workers_case_2(
    datasets,
    algorithm_request_case_2,
    workers_case_2,
    algorithm_data_loader_case_2,
    command_id_generator,
    controller,
):
    algorithm_request_dto = algorithm_request_case_2[1]
    workers = workers_case_2

    data_model_views_creator = DataModelViewsCreator(
        local_workers=workers.local_workers,
        variable_groups=algorithm_data_loader_case_2.get_variable_groups(),
        var_filters=algorithm_request_dto.inputdata.filters,
        dropna=algorithm_data_loader_case_2.get_dropna(),
        check_min_rows=algorithm_data_loader_case_2.get_check_min_rows(),
        command_id=command_id_generator.get_next_command_id(),
    )
    data_model_views_creator.create_data_model_views()
    data_model_views = data_model_views_creator.data_model_views

    local_workers_filtered = (
        data_model_views_creator.data_model_views.get_list_of_workers()
    )
    if not local_workers_filtered:
        pytest.fail(
            f"None of the workers contains data to execute the request: {algorithm_request_dto=}"
        )

    workers = Workers(
        global_worker=workers.global_worker, local_workers=local_workers_filtered
    )
    return (data_model_views, workers)


@pytest.fixture(scope="function")
def engine_case_1(
    algorithm_request_case_1,
    context_id,
    data_model_views_and_workers_case_1,
    command_id_generator,
):
    algorithm_request_dto = algorithm_request_case_1[1]

    engine_init_params = EngineInitParams(
        smpc_params=SMPCParams(smpc_enabled=False, smpc_optional=False),
        request_id=algorithm_request_dto.request_id,
        algo_flags=algorithm_request_dto.flags,
    )
    return _create_algorithm_execution_engine(
        engine_init_params=engine_init_params,
        command_id_generator=command_id_generator,
        workers=data_model_views_and_workers_case_1[1],
    )


@pytest.fixture(scope="function")
def engine_case_2(
    algorithm_request_case_2,
    context_id,
    data_model_views_and_workers_case_2,
    command_id_generator,
):
    algorithm_request_dto = algorithm_request_case_2[1]

    engine_init_params = EngineInitParams(
        smpc_params=SMPCParams(smpc_enabled=False, smpc_optional=False),
        request_id=algorithm_request_dto.request_id,
        algo_flags=algorithm_request_dto.flags,
    )
    return _create_algorithm_execution_engine(
        engine_init_params=engine_init_params,
        command_id_generator=command_id_generator,
        workers=data_model_views_and_workers_case_2[1],
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "algorithm,data_model_views_and_workers,metadata",
    [
        (
            "algorithm_case_1",
            "data_model_views_and_workers_case_1",
            "metadata_case_1",
        ),
        (
            "algorithm_case_2",
            "data_model_views_and_workers_case_2",
            "metadata_case_2",
        ),
    ],
)
@pytest.mark.asyncio
async def test_single_local_worker_algorithm_execution(
    algorithm,
    data_model_views_and_workers,
    metadata,
    controller,
    request,
    reset_celery_app_factory,  # celery celery fail if this is not reset
):
    algorithm = request.getfixturevalue(algorithm)
    data_model_views = request.getfixturevalue(data_model_views_and_workers)
    metadata = request.getfixturevalue(metadata)
    try:
        algorithm_result = await _algorithm_run_in_event_loop(
            algorithm=algorithm,
            data_model_views=data_model_views[0],
            metadata=metadata,
        )
    except Exception as exc:
        pytest.fail(f"Execution of the algorithm failed with {exc=}")

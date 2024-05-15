import time
from datetime import datetime
from datetime import timedelta
from os import path
from unittest.mock import patch

import pytest
from freezegun import freeze_time

from exareme2 import AttrDict
from exareme2 import algorithm_classes
from exareme2.algorithms.exareme2.algorithm import (
    InitializationParams as AlgorithmInitParams,
)
from exareme2.algorithms.exareme2.algorithm import Variables
from exareme2.controller import logger as ctrl_logger
from exareme2.controller.services.api.algorithm_request_dtos import (
    AlgorithmInputDataDTO,
)
from exareme2.controller.services.api.algorithm_request_dtos import AlgorithmRequestDTO
from exareme2.controller.services.exareme2.cleaner import Cleaner
from exareme2.controller.services.exareme2.controller import Controller
from exareme2.controller.services.exareme2.controller import DataModelViewsCreator
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
from tests.standalone_tests.conftest import ALGORITHM_FOLDERS_ENV_VARIABLE_VALUE
from tests.standalone_tests.conftest import (
    CONTROLLER_GLOBALWORKER_LOCALWORKER1_LOCALWORKER2_LOCALWORKERTMP_ADDRESSES_FILE,
)
from tests.standalone_tests.conftest import LOCALWORKERTMP_CONFIG_FILE
from tests.standalone_tests.conftest import RABBITMQ_LOCALWORKERTMP_NAME
from tests.standalone_tests.conftest import RABBITMQ_LOCALWORKERTMP_PORT
from tests.standalone_tests.conftest import TEST_ENV_CONFIG_FOLDER
from tests.standalone_tests.conftest import _create_rabbitmq_container
from tests.standalone_tests.conftest import _create_worker_service
from tests.standalone_tests.conftest import kill_service
from tests.standalone_tests.conftest import remove_localworkertmp_rabbitmq

WAIT_CLEANUP_TIME_LIMIT = 120
wla_WAIT_TIME_LIMIT = 120


@pytest.fixture
def controller_config():
    controller_config = {
        "log_level": "DEBUG",
        "framework_log_level": "INFO",
        "deployment_type": "LOCAL",
        "worker_landscape_aggregator_update_interval": 30,
        "flower_execution_timeout": 30,
        "cleanup": {
            "contextids_cleanup_folder": "/tmp/test_cleanup_entries",
            "workers_cleanup_interval": 2,
            "contextid_release_timelimit": 3600,  # 1hour
        },
        "localworkers": {
            "config_file": path.join(
                TEST_ENV_CONFIG_FOLDER,
                CONTROLLER_GLOBALWORKER_LOCALWORKER1_LOCALWORKER2_LOCALWORKERTMP_ADDRESSES_FILE,
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


@pytest.fixture(autouse=True, scope="session")
def init_background_controller_logger():
    ctrl_logger.set_background_service_logger("DEBUG")


@pytest.fixture(scope="function")
def controller(controller_config, cleaner, worker_landscape_aggregator):
    controller_config = AttrDict(controller_config)

    controller = Controller(
        worker_landscape_aggregator=worker_landscape_aggregator,
        cleaner=cleaner,
        logger=ctrl_logger.get_background_service_logger(),
        tasks_timeout=controller_config.rabbitmq.celery_tasks_timeout,
        run_udf_task_timeout=controller_config.rabbitmq.celery_run_udf_task_timeout,
        smpc_params=SMPCParams(
            smpc_enabled=False,
            smpc_optional=False,
        ),
    )
    return controller


@pytest.fixture(autouse=True)
def patch_celery_app(controller_config):
    with patch(
        "exareme2.controller.celery.app.controller_config",
        AttrDict(controller_config),
    ):
        yield


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
def algorithm_request_dto(datasets):
    return AlgorithmRequestDTO(
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
        parameters={"positive_class": "AD"},
    )


@pytest.fixture(scope="function")
def context_id():
    return UIDGenerator().get_a_uid()


@pytest.fixture(scope="function")
def algorithm_name():
    return "logistic_regression"


@pytest.fixture(scope="function")
def worker_landscape_aggregator(
    controller_config,
    globalworker_worker_service,
    localworker1_worker_service,
    localworker2_worker_service,
    localworkertmp_worker_service,
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


@pytest.fixture(scope="function")
def workers(controller, context_id, algorithm_request_dto):
    return controller._create_workers(
        request_id=algorithm_request_dto.request_id,
        context_id=context_id,
        data_model=algorithm_request_dto.inputdata.data_model,
        datasets=algorithm_request_dto.inputdata.datasets,
    )


@pytest.fixture(scope="function")
def data_model_views_and_workers(
    controller,
    datasets,
    algorithm_request_dto,
    workers,
    algorithm,
    command_id_generator,
):
    data_model_views_creator = DataModelViewsCreator(
        workers=workers.local_workers,
        variable_groups=algorithm.get_variable_groups(),
        var_filters=algorithm_request_dto.inputdata.filters,
        dropna=algorithm.get_dropna(),
        check_min_rows=algorithm.get_check_min_rows(),
        command_id=command_id_generator.get_next_command_id(),
    )
    data_model_views_creator.create_data_model_views()
    data_model_views = data_model_views_creator.data_model_views

    local_workers_filtered = (
        data_model_views_creator.data_model_views.get_list_of_workers()
    )
    workers = Workers(
        global_worker=workers.global_worker, local_workers=local_workers_filtered
    )
    return (data_model_views, workers)


@pytest.fixture(scope="function")
def metadata(worker_landscape_aggregator, algorithm_request_dto):
    variable_names = (algorithm_request_dto.inputdata.x or []) + (
        algorithm_request_dto.inputdata.y or []
    )
    return worker_landscape_aggregator.get_metadata(
        data_model=algorithm_request_dto.inputdata.data_model,
        variable_names=variable_names,
    )


@pytest.fixture(scope="function")
def algorithm(algorithm_request_dto, metadata):
    algorithm_name = "logistic_regression"
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
        metadata=metadata,
    )
    return algorithm_classes[algorithm_name](initialization_params=init_params)


@pytest.fixture(scope="function")
def command_id_generator():
    return CommandIdGenerator()


@pytest.fixture(scope="function")
def engine(
    algorithm_request_dto,
    context_id,
    data_model_views_and_workers,
    command_id_generator,
):
    engine_init_params = EngineInitParams(
        smpc_enabled=False,
        smpc_optional=False,
        request_id=algorithm_request_dto.request_id,
        context_id=context_id,
        algo_flags=algorithm_request_dto.flags,
        data_model_views=data_model_views_and_workers[0],  # data_model_views
    )
    return _create_algorithm_execution_engine(
        engine_init_params=engine_init_params,
        command_id_generator=command_id_generator,
        workers=data_model_views_and_workers[1],  # workers,
    )


@pytest.fixture(scope="function")
def cleaner(controller_config, worker_landscape_aggregator):
    controller_config = AttrDict(controller_config)

    cleaner = Cleaner(
        logger=ctrl_logger.get_background_service_logger(),
        cleanup_interval=controller_config.cleanup.workers_cleanup_interval,
        contextid_release_timelimit=controller_config.cleanup.contextid_release_timelimit,
        cleanup_task_timeout=controller_config.rabbitmq.celery_cleanup_task_timeout,
        run_udf_task_timeout=controller_config.rabbitmq.celery_run_udf_task_timeout,
        contextids_cleanup_folder=controller_config.cleanup.contextids_cleanup_folder,
        worker_landscape_aggregator=worker_landscape_aggregator,
    )

    return cleaner


@pytest.fixture
def db_cursors(
    monetdb_globalworker,
    monetdb_localworker1,
    monetdb_localworker2,
    monetdb_localworkertmp,
    globalworker_db_cursor,
    localworker1_db_cursor,
    localworker2_db_cursor,
    localworkertmp_db_cursor,
):
    # TODO the worker_ids should not be hardcoded..
    return {
        "testglobalworker": globalworker_db_cursor,
        "testlocalworker1": localworker1_db_cursor,
        "testlocalworker2": localworker2_db_cursor,
        "testlocalworkertmp": localworkertmp_db_cursor,
    }


@pytest.mark.slow
@pytest.mark.very_slow
def test_synchronous_cleanup(
    context_id,
    cleaner,
    worker_landscape_aggregator,
    reset_celery_app_factory,  # celery fail if this is not reset
    db_cursors,
):
    # Cleaner gets info about the workers via the WorkerLandscapeAggregator
    # Poll WorkerLandscapeAggregator until it has some worker info
    wait_wla(worker_landscape_aggregator)

    cleaner._reset()  # deletes all existing persistence files (cleanup_<context_id>.toml files)

    # contextid is added to Cleaner but is not released
    cleaner.add_contextid_for_cleanup(
        context_id, [worker_id for worker_id in db_cursors.keys()]
    )

    # create some dummy tables
    for worker_id, cursor in db_cursors.items():
        create_dummy_tables(worker_id, cursor, context_id)

    tables_before_cleanup = {
        worker_id: get_tables(cursor, context_id)
        for worker_id, cursor in db_cursors.items()
    }
    # check tables were created on all workers
    for worker_id, tables in tables_before_cleanup.items():
        if not tables:
            pytest.fail(
                f"{worker_id=} did not create any tables during the algorithm execution"
            )

    cleaner.cleanup_context_id(context_id=context_id)
    tables_after_cleanup = {
        worker_id: get_tables(cursor, context_id)
        for worker_id, cursor in db_cursors.items()
    }

    start = time.time()
    while any(tables_after_cleanup.values()):
        cleaner.cleanup_context_id(context_id=context_id)
        tables_after_cleanup = {
            worker_id: get_tables(cursor, context_id)
            for worker_id, cursor in db_cursors.items()
        }

        now = time.time()
        if now - start > WAIT_CLEANUP_TIME_LIMIT:
            pytest.fail(
                f"Some of the workers were not cleaned during {WAIT_CLEANUP_TIME_LIMIT=}\n"
                f"{tables_after_cleanup=}"
            )
        time.sleep(0.5)


@pytest.mark.slow
@pytest.mark.very_slow
def test_asynchronous_cleanup(
    context_id,
    cleaner,
    worker_landscape_aggregator,
    reset_celery_app_factory,  # celery fail if this is not reset
    db_cursors,
):
    # Cleaner gets info about the workers via the WorkerLandscapeAggregator
    # Poll WorkerLandscapeAggregator until it has some worker info
    wait_wla(worker_landscape_aggregator)

    # Start the Cleaner
    cleaner._reset()  # deletes all existing persistence files (cleanup_<context_id>.toml files)
    cleaner.start()

    # contextid is added to Cleaner but is not yet released
    cleaner.add_contextid_for_cleanup(
        context_id, [worker_id for worker_id in db_cursors.keys()]
    )

    # create some dummy tables
    for worker_id, cursor in db_cursors.items():
        create_dummy_tables(worker_id, cursor, context_id)

    tables_before_cleanup = {
        worker_id: get_tables(cursor, context_id)
        for worker_id, cursor in db_cursors.items()
    }
    # check tables were created on all workers
    for worker_id, tables in tables_before_cleanup.items():
        if not tables:
            pytest.fail(
                f"{worker_id=} did not create any tables during the algorithm execution"
            )

    # Releasing contextid, allows the Cleaner to schedule cleaning the contextid from
    # the workers in its next iteration (check Cleaner._cleanup_loop())
    cleaner.release_context_id(context_id=context_id)

    tables_after_cleanup = {
        worker_id: get_tables(cursor, context_id)
        for worker_id, cursor in db_cursors.items()
    }
    start = time.time()
    while any(tables_after_cleanup.values()):
        tables_after_cleanup = {
            worker_id: get_tables(cursor, context_id)
            for worker_id, cursor in db_cursors.items()
        }
        now = time.time()
        if now - start > WAIT_CLEANUP_TIME_LIMIT:
            pytest.fail(
                f"Some of the workers were not cleaned during {WAIT_CLEANUP_TIME_LIMIT=}\n"
                f"{tables_after_cleanup=}"
            )
        time.sleep(0.5)

    cleaner.stop()
    assert True


@pytest.mark.slow
@pytest.mark.very_slow
def test_cleanup_triggered_by_release_timelimit(
    context_id,
    cleaner,
    worker_landscape_aggregator,
    reset_celery_app_factory,  # celery celery fail if this is not reset
    db_cursors,
    controller_config,
):
    # Cleaner gets info about the workers via the WorkerLandscapeAggregator
    # Poll WorkerLandscapeAggregator until it has some worker info
    wait_wla(worker_landscape_aggregator)

    # Start the Cleaner
    cleaner._reset()  # deletes all existing persistence files(cleanup files)
    cleaner.start()

    # create some dummy tables
    for worker_id, cursor in db_cursors.items():
        create_dummy_tables(worker_id, cursor, context_id)

    tables_before_cleanup = {
        worker_id: get_tables(cursor, context_id)
        for worker_id, cursor in db_cursors.items()
    }
    # check tables were created on all workers
    for worker_id, tables in tables_before_cleanup.items():
        if not tables:
            pytest.fail(
                f"{worker_id=} did not create any tables during the algorithm execution"
            )

    # contextid is added to Cleaner but is not released
    cleaner.add_contextid_for_cleanup(
        context_id, [worker_id for worker_id in db_cursors.keys()]
    )

    controller_config = AttrDict(controller_config)
    passed_release_time = datetime.now() + timedelta(
        seconds=controller_config.cleanup.contextid_release_timelimit
    )
    with freeze_time(passed_release_time):
        tables_after_cleanup = {
            worker_id: get_tables(cursor, context_id)
            for worker_id, cursor in db_cursors.items()
        }
        start = time.time()
        while any(tables_after_cleanup.values()):
            tables_after_cleanup = {
                worker_id: get_tables(cursor, context_id)
                for worker_id, cursor in db_cursors.items()
            }
            now = time.time()
            if now - start > WAIT_CLEANUP_TIME_LIMIT:
                pytest.fail(
                    f"Some of the workers were not cleaned during {WAIT_CLEANUP_TIME_LIMIT=}\n"
                    f"{tables_after_cleanup=}"
                )
            time.sleep(0.5)

    cleaner.stop()
    assert True


@pytest.mark.slow
@pytest.mark.very_slow
def test_cleanup_after_rabbitmq_restart(
    localworkertmp_worker_service,
    context_id,
    cleaner,
    worker_landscape_aggregator,
    reset_celery_app_factory,  # celery fail if this is not reset
    db_cursors,
):
    # Cleaner gets info about the workers via the WorkerLandscapeAggregator
    # Poll WorkerLandscapeAggregator until it has some worker info
    wait_wla(worker_landscape_aggregator)

    # Start the Cleaner
    cleaner._reset()  # deletes all existing persistence files(cleanup files)
    cleaner.start()

    # Add contextid to Cleaner but is not yet released
    cleaner.add_contextid_for_cleanup(
        context_id,
        [worker_id for worker_id in db_cursors.keys()],
    )

    # create some dummy tables
    for worker_id, cursor in db_cursors.items():
        create_dummy_tables(worker_id, cursor, context_id)

    tables_before_cleanup = {
        worker_id: get_tables(cursor, context_id)
        for worker_id, cursor in db_cursors.items()
    }
    # check tables were created on all workers
    for worker_id, tables in tables_before_cleanup.items():
        if not tables:
            pytest.fail(
                f"{worker_id=} did not create any tables during the algorithm execution"
            )

    # kill rabbitmq container for localworkertmp
    remove_localworkertmp_rabbitmq()
    # kill the celery app of localworkertmp
    kill_service(localworkertmp_worker_service)

    # Releasing contextid, allows the Cleaner to schedule cleaning the contextid from
    # the workers in its next iteration (Cleaner._cleanup_loop())
    # Nevertheless, localworkertmp is currently down, so cannot be cleaned until it gets
    # back up
    cleaner.release_context_id(context_id=context_id)

    # restart tmplocalworker rabbitmq container
    _create_rabbitmq_container(
        RABBITMQ_LOCALWORKERTMP_NAME, RABBITMQ_LOCALWORKERTMP_PORT
    )

    # restart the celery app of localworkertmp
    localworkertmp_worker_service_proc = start_localworkertmp_worker_service()

    tables_after_cleanup = {
        worker_id: get_tables(cursor, context_id)
        for worker_id, cursor in db_cursors.items()
    }
    start = time.time()
    while any(tables_after_cleanup.values()):
        tables_after_cleanup = {
            worker_id: get_tables(cursor, context_id)
            for worker_id, cursor in db_cursors.items()
        }
        now = time.time()
        if now - start > WAIT_CLEANUP_TIME_LIMIT:
            pytest.fail(
                f"Some of the workers were not cleaned during {WAIT_CLEANUP_TIME_LIMIT=}\n"
                f"{tables_after_cleanup=}"
            )
        time.sleep(0.5)

    cleaner.stop()

    # the worker service was started in here so it must manually killed, otherwise it is
    # alive through the whole pytest session and is erroneously accessed by other tests
    # where the worker service is supposedly down
    kill_service(localworkertmp_worker_service_proc)

    assert True


@pytest.mark.slow
@pytest.mark.very_slow
def test_cleanup_after_worker_service_restart(
    localworkertmp_worker_service,
    context_id,
    cleaner,
    worker_landscape_aggregator,
    reset_celery_app_factory,  # celery celery fail if this is not reset
    db_cursors,
):
    # Cleaner gets info about the workers via the WorkerLandscapeAggregator
    # Poll WorkerLandscapeAggregator until it has some worker info
    wait_wla(worker_landscape_aggregator)

    # Start the Cleaner
    cleaner._reset()  # deletes all existing persistence files(cleanup files)
    cleaner.start()

    # Add contextid to Cleaner but is not yet released
    cleaner.add_contextid_for_cleanup(
        context_id, [worker_id for worker_id in db_cursors.keys()]
    )

    # create some dummy tables
    for worker_id, cursor in db_cursors.items():
        create_dummy_tables(worker_id, cursor, context_id)

    tables_before_cleanup = {
        worker_id: get_tables(cursor, context_id)
        for worker_id, cursor in db_cursors.items()
    }
    # check tables were created on all workers
    for worker_id, tables in tables_before_cleanup.items():
        if not tables:
            pytest.fail(
                f"{worker_id=} did not create any tables during the algorithm execution"
            )

    # kill the celery app of localworkertmp
    kill_service(localworkertmp_worker_service)

    # Releasing contextid, allows the Cleaner to schedule cleaning the contextid from
    # the workers in its next iteration (Cleaner._cleanup_loop())
    # Nevertheless, localworkertmp is currently down, so cannot be cleaned until it gets
    # back up
    cleaner.release_context_id(context_id=context_id)

    # restart tmplocalworker worker service (the celery app)
    localworkertmp_worker_service_proc = start_localworkertmp_worker_service()

    tables_after_cleanup = {
        worker_id: get_tables(cursor, context_id)
        for worker_id, cursor in db_cursors.items()
    }
    start = time.time()
    while any(tables_after_cleanup.values()):
        tables_after_cleanup = {
            worker_id: get_tables(cursor, context_id)
            for worker_id, cursor in db_cursors.items()
        }

        now = time.time()
        if now - start > WAIT_CLEANUP_TIME_LIMIT:
            pytest.fail(
                f"Some of the workers were not cleaned during {WAIT_CLEANUP_TIME_LIMIT=}\n"
                f"{tables_after_cleanup=}"
            )
        time.sleep(0.5)

    cleaner.stop()

    # the worker service was started in here so it must manually killed, otherwise it is
    # alive through the whole pytest session and is erroneously accessed by other tests
    # where teh worker service is supposedly down
    kill_service(localworkertmp_worker_service_proc)

    assert True


def start_localworkertmp_worker_service():
    worker_config_file = LOCALWORKERTMP_CONFIG_FILE
    algo_folders_env_variable_val = ALGORITHM_FOLDERS_ENV_VARIABLE_VALUE
    worker_config_filepath = path.join(TEST_ENV_CONFIG_FOLDER, worker_config_file)
    proc = _create_worker_service(algo_folders_env_variable_val, worker_config_filepath)
    return proc


def flatten_list(l: list):
    return [item for sub_l in l for item in sub_l]


def create_dummy_tables(worker_id, cursor, context_id):
    columns = '"col1" INT,"col2" DOUBLE'
    query = ""
    for i in range(10):
        table_name = f"normal_{worker_id}_{context_id}_0_{i}"
        query = query + f"CREATE TABLE {table_name}({columns});"
    cursor.execute(query)


def get_tables(cursor, context_id):
    query = f"""
    SELECT name FROM tables
    WHERE name LIKE '%{context_id.lower()}%'
    AND system=FALSE;
    """
    result = cursor.execute(query).fetchall()
    return [i[0] for i in result]


def wait_wla(worker_landscape_aggregator):
    start = time.time()
    while (
        not worker_landscape_aggregator.get_workers()
        or not worker_landscape_aggregator.get_cdes_per_data_model()
        or not worker_landscape_aggregator.get_datasets_locations()
    ):
        if time.time() - start > wla_WAIT_TIME_LIMIT:
            pytest.fail(
                "Exceeded max retries while waiting for the worker landscape aggregator to"
                "return some workers"
            )
        time.sleep(0.5)

import time
from datetime import datetime
from datetime import timedelta
from os import path
from unittest.mock import patch

import pytest
from freezegun import freeze_time

from mipengine import AttrDict
from mipengine import algorithm_classes
from mipengine.algorithms.algorithm import InitializationParams as AlgorithmInitParams
from mipengine.algorithms.algorithm import Variables
from mipengine.controller import controller_logger as ctrl_logger
from mipengine.controller.algorithm_execution_engine import (
    InitializationParams as EngineInitParams,
)
from mipengine.controller.api.algorithm_request_dto import AlgorithmInputDataDTO
from mipengine.controller.api.algorithm_request_dto import AlgorithmRequestDTO
from mipengine.controller.cleaner import Cleaner
from mipengine.controller.cleaner import InitializationParams as CleanerInitParams
from mipengine.controller.controller import CommandIdGenerator
from mipengine.controller.controller import Controller
from mipengine.controller.controller import InitializationParams as ControllerInitParams
from mipengine.controller.controller import Nodes
from mipengine.controller.controller import _create_algorithm_execution_engine
from mipengine.controller.controller import _get_data_model_views_nodes
from mipengine.controller.controller import sanitize_request_variable
from mipengine.controller.node_landscape_aggregator import (
    InitializationParams as NodeLandscapeAggregatorInitParams,
)
from mipengine.controller.node_landscape_aggregator import NodeLandscapeAggregator
from mipengine.controller.uid_generator import UIDGenerator
from tests.standalone_tests.conftest import ALGORITHM_FOLDERS_ENV_VARIABLE_VALUE
from tests.standalone_tests.conftest import (
    CONTROLLER_GLOBALNODE_LOCALNODE1_LOCALNODE2_LOCALNODETMP_ADDRESSES_FILE,
)
from tests.standalone_tests.conftest import LOCALNODETMP_CONFIG_FILE
from tests.standalone_tests.conftest import RABBITMQ_LOCALNODETMP_NAME
from tests.standalone_tests.conftest import RABBITMQ_LOCALNODETMP_PORT
from tests.standalone_tests.conftest import TEST_ENV_CONFIG_FOLDER
from tests.standalone_tests.conftest import _create_node_service
from tests.standalone_tests.conftest import _create_rabbitmq_container
from tests.standalone_tests.conftest import kill_service
from tests.standalone_tests.conftest import remove_localnodetmp_rabbitmq

WAIT_CLEANUP_TIME_LIMIT = 60
WAIT_BEFORE_BRING_TMPNODE_DOWN = 60
NLA_WAIT_TIME_LIMIT = 120


@pytest.fixture
def controller_config():
    controller_config = {
        "log_level": "DEBUG",
        "framework_log_level": "INFO",
        "deployment_type": "LOCAL",
        "node_landscape_aggregator_update_interval": 30,
        "cleanup": {
            "contextids_cleanup_folder": "/tmp/test_cleanup_entries",
            "nodes_cleanup_interval": 2,
            "contextid_release_timelimit": 3600,  # 1hour
        },
        "localnodes": {
            "config_file": path.join(
                TEST_ENV_CONFIG_FOLDER,
                CONTROLLER_GLOBALNODE_LOCALNODE1_LOCALNODE2_LOCALNODETMP_ADDRESSES_FILE,
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
def controller(controller_config, cleaner, node_landscape_aggregator):
    controller_config = AttrDict(controller_config)

    controller_init_params = ControllerInitParams(
        smpc_enabled=False,
        smpc_optional=False,
        celery_tasks_timeout=controller_config.rabbitmq.celery_tasks_timeout,
        celery_run_udf_task_timeout=controller_config.rabbitmq.celery_run_udf_task_timeout,
    )
    controller = Controller(
        initialization_params=controller_init_params,
        cleaner=cleaner,
        node_landscape_aggregator=node_landscape_aggregator,
    )
    return controller


@pytest.fixture(autouse=True)
def patch_celery_app(controller_config):
    with patch(
        "mipengine.controller.celery_app.controller_config",
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
def node_landscape_aggregator(
    controller_config,
    globalnode_node_service,
    localnode1_node_service,
    localnode2_node_service,
    localnodetmp_node_service,
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

    # TODO https://team-1617704806227.atlassian.net/jira/software/projects/MIP/issues/MIP-771
    yield node_landscape_aggregator
    del node_landscape_aggregator


@pytest.fixture(scope="function")
def nodes(controller, context_id, algorithm_request_dto):
    return controller._create_nodes(
        request_id=algorithm_request_dto.request_id,
        context_id=context_id,
        data_model=algorithm_request_dto.inputdata.data_model,
        datasets=algorithm_request_dto.inputdata.datasets,
    )


@pytest.fixture(scope="function")
def data_model_views_and_nodes(
    controller, datasets, algorithm_request_dto, nodes, algorithm, command_id_generator
):
    data_model_views = controller._create_data_model_views(
        local_nodes=nodes.local_nodes,
        datasets=datasets,
        data_model=algorithm_request_dto.inputdata.data_model,
        variable_groups=algorithm.get_variable_groups(),
        var_filters=algorithm_request_dto.inputdata.filters,
        dropna=algorithm.get_dropna(),
        check_min_rows=algorithm.get_check_min_rows(),
        command_id=command_id_generator.get_next_command_id(),
    )
    local_nodes_filtered = _get_data_model_views_nodes(data_model_views)
    nodes = Nodes(global_node=nodes.global_node, local_nodes=local_nodes_filtered)
    return (data_model_views, nodes)


@pytest.fixture(scope="function")
def metadata(node_landscape_aggregator, algorithm_request_dto):
    variable_names = (algorithm_request_dto.inputdata.x or []) + (
        algorithm_request_dto.inputdata.y or []
    )
    return node_landscape_aggregator.get_metadata(
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
    algorithm_request_dto, context_id, data_model_views_and_nodes, command_id_generator
):
    engine_init_params = EngineInitParams(
        smpc_enabled=False,
        smpc_optional=False,
        request_id=algorithm_request_dto.request_id,
        context_id=context_id,
        algo_flags=algorithm_request_dto.flags,
        data_model_views=data_model_views_and_nodes[0],  # data_model_views
    )
    return _create_algorithm_execution_engine(
        engine_init_params=engine_init_params,
        command_id_generator=command_id_generator,
        nodes=data_model_views_and_nodes[1],  # nodes,
    )


@pytest.fixture(scope="function")
def cleaner(controller_config, node_landscape_aggregator):
    controller_config = AttrDict(controller_config)
    cleaner_init_params = CleanerInitParams(
        cleanup_interval=controller_config.cleanup.nodes_cleanup_interval,
        contextid_release_timelimit=controller_config.cleanup.contextid_release_timelimit,
        celery_cleanup_task_timeout=controller_config.rabbitmq.celery_cleanup_task_timeout,
        celery_run_udf_task_timeout=controller_config.rabbitmq.celery_run_udf_task_timeout,
        contextids_cleanup_folder=controller_config.cleanup.contextids_cleanup_folder,
        node_landscape_aggregator=node_landscape_aggregator,
    )
    cleaner = Cleaner(cleaner_init_params)
    yield cleaner
    del cleaner


@pytest.fixture
def db_cursors(
    monetdb_globalnode,
    monetdb_localnode1,
    monetdb_localnode2,
    monetdb_localnodetmp,
    globalnode_db_cursor,
    localnode1_db_cursor,
    localnode2_db_cursor,
    localnodetmp_db_cursor,
):
    # TODO the node_ids should not be hardcoded..
    return {
        "testglobalnode": globalnode_db_cursor,
        "testlocalnode1": localnode1_db_cursor,
        "testlocalnode2": localnode2_db_cursor,
        "testlocalnodetmp": localnodetmp_db_cursor,
    }


@pytest.mark.slow
@pytest.mark.very_slow
def test_synchronous_cleanup(
    context_id,
    cleaner,
    node_landscape_aggregator,
    reset_celery_app_factory,  # celery tasks fail if this is not reset
    db_cursors,
):

    # Cleaner gets info about the nodes via the NodeLandscapeAggregator
    # Poll NodeLandscapeAggregator until it has some node info
    wait_nla(node_landscape_aggregator)

    cleaner._reset()  # deletes all existing persistence files (cleanup_<context_id>.toml files)

    # contextid is added to Cleaner but is not released
    cleaner.add_contextid_for_cleanup(
        context_id, [node_id for node_id in db_cursors.keys()]
    )

    # create some dummy tables
    for node_id, cursor in db_cursors.items():
        create_dummy_tables(node_id, cursor, context_id)

    tables_before_cleanup = {
        node_id: get_tables(cursor, context_id)
        for node_id, cursor in db_cursors.items()
    }
    # check tables were created on all nodes
    for node_id, tables in tables_before_cleanup.items():
        if not tables:
            pytest.fail(
                f"{node_id=} did not create any tables during the algorithm execution"
            )

    cleaner.cleanup_context_id(context_id=context_id)
    tables_after_cleanup = {
        node_id: get_tables(cursor, context_id)
        for node_id, cursor in db_cursors.items()
    }

    start = time.time()
    while not all(
        tables == [] for tables in flatten_list(tables_after_cleanup.values())
    ):

        cleaner.cleanup_context_id(context_id=context_id)
        tables_after_cleanup = {
            node_id: get_tables(cursor, context_id)
            for node_id, cursor in db_cursors.items()
        }

        now = time.time()
        if now - start > WAIT_CLEANUP_TIME_LIMIT:
            pytest.fail(
                f"Some of the nodes were not cleaned during {WAIT_CLEANUP_TIME_LIMIT=}\n"
                f"{tables_after_cleanup=}"
            )
        time.sleep(0.5)


@pytest.mark.slow
@pytest.mark.very_slow
def test_asynchronous_cleanup(
    context_id,
    cleaner,
    node_landscape_aggregator,
    reset_celery_app_factory,  # celery tasks fail if this is not reset
    db_cursors,
):

    # Cleaner gets info about the nodes via the NodeLandscapeAggregator
    # Poll NodeLandscapeAggregator until it has some node info
    wait_nla(node_landscape_aggregator)

    # Start the Cleaner
    cleaner._reset()  # deletes all existing persistence files (cleanup_<context_id>.toml files)
    cleaner.start()

    # contextid is added to Cleaner but is not yet released
    cleaner.add_contextid_for_cleanup(
        context_id, [node_id for node_id in db_cursors.keys()]
    )

    # create some dummy tables
    for node_id, cursor in db_cursors.items():
        create_dummy_tables(node_id, cursor, context_id)

    tables_before_cleanup = {
        node_id: get_tables(cursor, context_id)
        for node_id, cursor in db_cursors.items()
    }
    # check tables were created on all nodes
    for node_id, tables in tables_before_cleanup.items():
        if not tables:
            pytest.fail(
                f"{node_id=} did not create any tables during the algorithm execution"
            )

    # Releasing contextid, allows the Cleaner to schedule cleaning the contextid from
    # the nodes in its next iteration (check Cleaner._cleanup_loop())
    cleaner.release_context_id(context_id=context_id)

    tables_after_cleanup = {
        node_id: get_tables(cursor, context_id)
        for node_id, cursor in db_cursors.items()
    }
    start = time.time()
    while not all(
        (tables == [] for tables in flatten_list(tables_after_cleanup.values()))
    ):
        tables_after_cleanup = {
            node_id: get_tables(cursor, context_id)
            for node_id, cursor in db_cursors.items()
        }
        now = time.time()
        if now - start > WAIT_CLEANUP_TIME_LIMIT:
            pytest.fail(
                f"Some of the nodes were not cleaned during {WAIT_CLEANUP_TIME_LIMIT=}\n"
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
    node_landscape_aggregator,
    reset_celery_app_factory,  # celery tasks fail if this is not reset
    db_cursors,
    controller_config,
):
    # Cleaner gets info about the nodes via the NodeLandscapeAggregator
    # Poll NodeLandscapeAggregator until it has some node info
    wait_nla(node_landscape_aggregator)

    # Start the Cleaner
    cleaner._reset()  # deletes all existing persistence files(cleanup files)
    cleaner.start()

    # create some dummy tables
    for node_id, cursor in db_cursors.items():
        create_dummy_tables(node_id, cursor, context_id)

    tables_before_cleanup = {
        node_id: get_tables(cursor, context_id)
        for node_id, cursor in db_cursors.items()
    }
    # check tables were created on all nodes
    for node_id, tables in tables_before_cleanup.items():
        if not tables:
            pytest.fail(
                f"{node_id=} did not create any tables during the algorithm execution"
            )

    # contextid is added to Cleaner but is not released
    cleaner.add_contextid_for_cleanup(
        context_id, [node_id for node_id in db_cursors.keys()]
    )

    controller_config = AttrDict(controller_config)
    passed_release_time = datetime.now() + timedelta(
        seconds=controller_config.cleanup.contextid_release_timelimit
    )
    with freeze_time(passed_release_time):
        tables_after_cleanup = {
            node_id: get_tables(cursor, context_id)
            for node_id, cursor in db_cursors.items()
        }
        start = time.time()
        while not all(
            (tables is None for tables in flatten_list(tables_after_cleanup.values()))
        ):
            tables_after_cleanup = {
                node_id: get_tables(cursor, context_id)
                for node_id, cursor in db_cursors.items()
            }
            now = time.time()
            if now - start > WAIT_CLEANUP_TIME_LIMIT:
                pytest.fail(
                    f"Some of the nodes were not cleaned during {WAIT_CLEANUP_TIME_LIMIT=}\n"
                    f"{tables_after_cleanup=}"
                )
            time.sleep(0.5)

    cleaner.stop()
    assert True


@pytest.mark.slow
@pytest.mark.very_slow
def test_cleanup_after_rabbitmq_restart(
    localnodetmp_node_service,
    context_id,
    cleaner,
    node_landscape_aggregator,
    reset_celery_app_factory,  # celery tasks fail if this is not reset
    db_cursors,
):

    # Cleaner gets info about the nodes via the NodeLandscapeAggregator
    # Poll NodeLandscapeAggregator until it has some node info
    wait_nla(node_landscape_aggregator)

    # Start the Cleaner
    cleaner._reset()  # deletes all existing persistence files(cleanup files)
    cleaner.start()

    # Add contextid to Cleaner but is not yet released
    cleaner.add_contextid_for_cleanup(
        context_id,
        [node_id for node_id in db_cursors.keys()],
    )

    # create some dummy tables
    for node_id, cursor in db_cursors.items():
        create_dummy_tables(node_id, cursor, context_id)

    tables_before_cleanup = {
        node_id: get_tables(cursor, context_id)
        for node_id, cursor in db_cursors.items()
    }
    # check tables were created on all nodes
    for node_id, tables in tables_before_cleanup.items():
        if not tables:
            pytest.fail(
                f"{node_id=} did not create any tables during the algorithm execution"
            )

    # kill rabbitmq container for localnotmp
    remove_localnodetmp_rabbitmq()
    # kill the celery app of localnodetmp
    kill_service(localnodetmp_node_service)

    # Releasing contextid, allows the Cleaner to schedule cleaning the contextid from
    # the nodes in its next iteration (Cleaner._cleanup_loop())
    # Nevertheless, localnodetmp is currently down, so cannot be cleaned until it gets
    # back up
    cleaner.release_context_id(context_id=context_id)

    # restart tmplocalnode rabbitmq container
    _create_rabbitmq_container(RABBITMQ_LOCALNODETMP_NAME, RABBITMQ_LOCALNODETMP_PORT)

    # restart the celery app of localnodetmp
    localnodetmp_node_service_proc = start_localnodetmp_node_service()

    tables_after_cleanup = {
        node_id: get_tables(cursor, context_id)
        for node_id, cursor in db_cursors.items()
    }
    start = time.time()
    while not all(
        (tables == [] for tables in flatten_list(tables_after_cleanup.values()))
    ):
        tables_after_cleanup = {
            node_id: get_tables(cursor, context_id)
            for node_id, cursor in db_cursors.items()
        }
        now = time.time()
        if now - start > WAIT_CLEANUP_TIME_LIMIT:
            pytest.fail(
                f"Some of the nodes were not cleaned during {WAIT_CLEANUP_TIME_LIMIT=}\n"
                f"{tables_after_cleanup=}"
            )
        time.sleep(0.5)

    cleaner.stop()

    # the node service was started in here so it must manually killed, otherwise it is
    # alive through the whole pytest session and is erroneously accessed by other tests
    # where theh node service is supposedly down
    kill_service(localnodetmp_node_service_proc)

    assert True


@pytest.mark.slow
@pytest.mark.very_slow
def test_cleanup_after_node_service_restart(
    localnodetmp_node_service,
    context_id,
    cleaner,
    node_landscape_aggregator,
    reset_celery_app_factory,  # celery tasks fail if this is not reset
    db_cursors,
):
    # Cleaner gets info about the nodes via the NodeLandscapeAggregator
    # Poll NodeLandscapeAggregator until it has some node info
    wait_nla(node_landscape_aggregator)

    # Start the Cleaner
    cleaner._reset()  # deletes all existing persistence files(cleanup files)
    cleaner.start()

    # Add contextid to Cleaner but is not yet released
    cleaner.add_contextid_for_cleanup(
        context_id, [node_id for node_id in db_cursors.keys()]
    )

    # create some dummy tables
    for node_id, cursor in db_cursors.items():
        create_dummy_tables(node_id, cursor, context_id)

    tables_before_cleanup = {
        node_id: get_tables(cursor, context_id)
        for node_id, cursor in db_cursors.items()
    }
    # check tables were created on all nodes
    for node_id, tables in tables_before_cleanup.items():
        if not tables:
            pytest.fail(
                f"{node_id=} did not create any tables during the algorithm execution"
            )

    # kill the celery app of localnodetmp
    kill_service(localnodetmp_node_service)

    # Releasing contextid, allows the Cleaner to schedule cleaning the contextid from
    # the nodes in its next iteration (Cleaner._cleanup_loop())
    # Nevertheless, localnodetmp is currently down, so cannot be cleaned until it gets
    # back up
    cleaner.release_context_id(context_id=context_id)

    # restart tmplocalnode node service (the celery app)
    localnodetmp_node_service_proc = start_localnodetmp_node_service()

    tables_after_cleanup = {
        node_id: get_tables(cursor, context_id)
        for node_id, cursor in db_cursors.items()
    }
    start = time.time()
    while not all(
        (tables == [] for tables in flatten_list(tables_after_cleanup.values()))
    ):

        tables_after_cleanup = {
            node_id: get_tables(cursor, context_id)
            for node_id, cursor in db_cursors.items()
        }

        now = time.time()
        if now - start > WAIT_CLEANUP_TIME_LIMIT:
            pytest.fail(
                f"Some of the nodes were not cleaned during {WAIT_CLEANUP_TIME_LIMIT=}\n"
                f"{tables_after_cleanup=}"
            )
        time.sleep(0.5)

    cleaner.stop()

    # the node service was started in here so it must manually killed, otherwise it is
    # alive through the whole pytest session and is erroneously accessed by other tests
    # where teh node service is supposedly down
    kill_service(localnodetmp_node_service_proc)

    assert True


def start_localnodetmp_node_service():
    node_config_file = LOCALNODETMP_CONFIG_FILE
    algo_folders_env_variable_val = ALGORITHM_FOLDERS_ENV_VARIABLE_VALUE
    node_config_filepath = path.join(TEST_ENV_CONFIG_FOLDER, node_config_file)
    proc = _create_node_service(algo_folders_env_variable_val, node_config_filepath)
    return proc


def flatten_list(l: list):
    return [item for sub_l in l for item in sub_l]


def create_dummy_tables(node_id, cursor, context_id):
    columns = '"col1" INT,"col2" DOUBLE'
    query = ""
    for i in range(10):
        table_name = f"normal_{node_id}_{context_id}_0_{i}"
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


def wait_nla(node_landscape_aggregator):
    start = time.time()
    while (
        not node_landscape_aggregator.get_nodes()
        or not node_landscape_aggregator.get_cdes_per_data_model()
        or not node_landscape_aggregator.get_datasets_locations()
    ):

        if time.time() - start > NLA_WAIT_TIME_LIMIT:
            pytest.fail(
                "Exceeded max retries while waiting for the node landscape aggregator to"
                "return some nodes"
            )
        time.sleep(0.5)

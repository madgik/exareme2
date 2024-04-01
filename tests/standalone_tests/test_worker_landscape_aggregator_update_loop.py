from os import path
from unittest.mock import patch

import pytest
import toml

from exareme2 import AttrDict
from exareme2.controller import logger as ctrl_logger
from exareme2.controller.services.worker_landscape_aggregator import (
    WorkerLandscapeAggregator,
)
from exareme2.controller.services.worker_landscape_aggregator import _NLARegistries
from exareme2.controller.workers_addresses import WorkersAddresses
from exareme2.controller.workers_addresses import WorkersAddressesFactory
from tests.standalone_tests.conftest import GLOBALWORKER_CONFIG_FILE
from tests.standalone_tests.conftest import LOCALWORKER1_CONFIG_FILE
from tests.standalone_tests.conftest import LOCALWORKER2_CONFIG_FILE
from tests.standalone_tests.conftest import LOCALWORKERTMP_CONFIG_FILE
from tests.standalone_tests.conftest import TEST_ENV_CONFIG_FOLDER


@pytest.fixture(scope="session")
def controller_config():
    controller_config = {
        "deployment_type": "LOCAL",
        "worker_landscape_aggregator_update_interval": 30,
        "localworkers": {
            "config_file": "./tests/standalone_tests/testing_env_configs/test_globalworker_localworker1_localworker2_localworkertmp_addresses.json"
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
        "smpc": {"enabled": False, "optional": False},
    }
    return controller_config


@pytest.fixture(autouse=True, scope="session")
def init_background_controller_logger():
    ctrl_logger.set_background_service_logger("DEBUG")


@pytest.fixture(autouse=True, scope="session")
def patch_celery_app(controller_config):
    with patch(
        "exareme2.controller.celery.app.controller_config", AttrDict(controller_config)
    ):
        yield


class CustomWorkerAddresses(WorkersAddresses):
    def __init__(self, socket_addresses):
        self._socket_addresses = socket_addresses


def get_custom_workers_addresses_without_workers() -> WorkersAddresses:
    return CustomWorkerAddresses([])


def get_custom_workers_addresses_global_and_tmp() -> WorkersAddresses:
    return CustomWorkerAddresses(["172.17.0.1:60000", "172.17.0.1:60003"])


def get_custom_workers_addresses() -> WorkersAddresses:
    return CustomWorkerAddresses(["172.17.0.1:60000", "172.17.0.1:60001"])


def get_custom_workers_addresses_1_2() -> WorkersAddresses:
    return CustomWorkerAddresses(
        ["172.17.0.1:60000", "172.17.0.1:60001", "172.17.0.1:60002"]
    )


@pytest.fixture(autouse=True, scope="function")
def patch_workers_addresses():
    with patch.object(
        WorkersAddressesFactory,
        "get_workers_addresses",
        side_effect=get_custom_workers_addresses,
    ) as patched:
        yield patched


@pytest.fixture(scope="function")
def worker_landscape_aggregator(controller_config):
    controller_config = AttrDict(controller_config)

    worker_landscape_aggregator = WorkerLandscapeAggregator(
        logger=ctrl_logger.get_background_service_logger(),
        update_interval=controller_config.worker_landscape_aggregator_update_interval,
        tasks_timeout=controller_config.rabbitmq.celery_tasks_timeout,
        run_udf_task_timeout=controller_config.rabbitmq.celery_run_udf_task_timeout,
        deployment_type=controller_config.deployment_type,
        localworkers=controller_config.localworkers,
    )
    worker_landscape_aggregator.stop()
    worker_landscape_aggregator.keep_updating = False
    worker_landscape_aggregator._nla_registries = _NLARegistries()

    return worker_landscape_aggregator


@pytest.mark.slow
def test_update_loop_data_properly_added(
    patch_workers_addresses,
    globalworker_worker_service,
    localworker1_worker_service,
    load_data_localworker1,
    worker_landscape_aggregator,
):
    worker_landscape_aggregator.update()

    assert (
        worker_landscape_aggregator.get_cdes_per_data_model().data_models_cdes
        and "tbi:0.1"
        in worker_landscape_aggregator.get_cdes_per_data_model().data_models_cdes
        and "dementia:0.1"
        in worker_landscape_aggregator.get_cdes_per_data_model().data_models_cdes
    )


@pytest.mark.slow
def test_update_loop_get_worker_info_fail(
    patch_workers_addresses, globalworker_worker_service, worker_landscape_aggregator
):
    patch_workers_addresses.side_effect = get_custom_workers_addresses_global_and_tmp
    worker_landscape_aggregator.update()
    assert worker_landscape_aggregator.get_workers()
    assert any(
        [
            worker.id == get_globalworker_worker_id()
            for worker in worker_landscape_aggregator.get_workers()
        ]
    )

    assert all(
        [
            worker.id != get_localworkertmp_worker_id()
            for worker in worker_landscape_aggregator.get_all_local_workers()
        ]
    )


@pytest.mark.slow
def test_update_loop_workers_properly_added(
    patch_workers_addresses,
    globalworker_worker_service,
    localworker1_worker_service,
    load_data_localworker1,
    localworker2_worker_service,
    load_data_localworker2,
    worker_landscape_aggregator,
):
    localworker1_worker_id = get_localworker1_worker_id()
    localworker2_worker_id = get_localworker2_worker_id()
    worker_landscape_aggregator.update()
    assert any(
        [
            worker.id == localworker1_worker_id
            for worker in worker_landscape_aggregator.get_all_local_workers()
        ]
    )

    assert all(
        [
            worker.id != localworker2_worker_id
            for worker in worker_landscape_aggregator.get_all_local_workers()
        ]
    )

    patch_workers_addresses.side_effect = get_custom_workers_addresses_1_2
    worker_landscape_aggregator.update()

    assert any(
        [
            worker.id == localworker2_worker_id
            for worker in worker_landscape_aggregator.get_all_local_workers()
        ]
    )


def get_localworker2_worker_id():
    local_worker_filepath = path.join(TEST_ENV_CONFIG_FOLDER, LOCALWORKER2_CONFIG_FILE)
    with open(local_worker_filepath) as fp:
        tmp = toml.load(fp)
        worker_id = tmp["identifier"]
    return worker_id


def get_localworker1_worker_id():
    local_worker_filepath = path.join(TEST_ENV_CONFIG_FOLDER, LOCALWORKER1_CONFIG_FILE)
    with open(local_worker_filepath) as fp:
        tmp = toml.load(fp)
        worker_id = tmp["identifier"]
    return worker_id


def get_localworkertmp_worker_id():
    local_worker_filepath = path.join(
        TEST_ENV_CONFIG_FOLDER, LOCALWORKERTMP_CONFIG_FILE
    )
    with open(local_worker_filepath) as fp:
        tmp = toml.load(fp)
        worker_id = tmp["identifier"]
    return worker_id


def get_globalworker_worker_id():
    local_worker_filepath = path.join(TEST_ENV_CONFIG_FOLDER, GLOBALWORKER_CONFIG_FILE)
    with open(local_worker_filepath) as fp:
        tmp = toml.load(fp)
        worker_id = tmp["identifier"]
    return worker_id

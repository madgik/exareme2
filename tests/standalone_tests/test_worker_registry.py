import pytest

from exareme2 import AttrDict
from exareme2.controller import logger as ctrl_logger
from exareme2.controller.services.worker_landscape_aggregator import DataModelRegistry
from exareme2.controller.services.worker_landscape_aggregator import (
    WorkerLandscapeAggregator,
)
from exareme2.controller.services.worker_landscape_aggregator import WorkerRegistry
from exareme2.worker_communication import WorkerInfo
from exareme2.worker_communication import WorkerRole

mocked_worker_addresses = [
    "127.0.0.1:5672",
    "127.0.0.2:5672",
    "127.0.0.3:5672",
    "127.0.0.4:5672",
]


@pytest.fixture(scope="function")
def mocked_nla():
    worker_landscape_aggregator = WorkerLandscapeAggregator(
        logger=ctrl_logger.get_background_service_logger(),
        update_interval=0,
        tasks_timeout=0,
        run_udf_task_timeout=0,
        deployment_type="",
        localworkers=AttrDict({}),
    )
    worker_landscape_aggregator.stop()
    worker_landscape_aggregator.keep_updating = False

    worker_registry = WorkerRegistry(
        workers_info=[
            WorkerInfo(
                id="globalworker",
                role=WorkerRole.GLOBALWORKER,
                ip=mocked_worker_addresses[0].split(":")[0],
                port=mocked_worker_addresses[0].split(":")[1],
                db_ip="127.0.0.1",
                db_port=50000,
            ),
            WorkerInfo(
                id="localworker1",
                role=WorkerRole.LOCALWORKER,
                ip=mocked_worker_addresses[1].split(":")[0],
                port=mocked_worker_addresses[1].split(":")[1],
                db_ip="127.0.0.1",
                db_port=50001,
            ),
            WorkerInfo(
                id="localworker2",
                role=WorkerRole.LOCALWORKER,
                ip=mocked_worker_addresses[2].split(":")[0],
                port=mocked_worker_addresses[2].split(":")[1],
                db_ip="127.0.0.1",
                db_port=50002,
            ),
            WorkerInfo(
                id="localworker3",
                role=WorkerRole.LOCALWORKER,
                ip=mocked_worker_addresses[2].split(":")[0],
                port=mocked_worker_addresses[2].split(":")[1],
                db_ip="127.0.0.1",
                db_port=50003,
            ),
        ]
    )
    worker_landscape_aggregator._set_new_registries(
        worker_registry=worker_registry, data_model_registry=DataModelRegistry()
    )
    return worker_landscape_aggregator


def test_get_workers(mocked_nla):
    workers = mocked_nla.get_workers()
    assert len(workers) == 4
    assert (
        len([worker for worker in workers if worker.role == WorkerRole.LOCALWORKER])
        == 3
    )
    assert (
        len([worker for worker in workers if worker.role == WorkerRole.GLOBALWORKER])
        == 1
    )


def test_get_global_worker(mocked_nla):
    global_worker = mocked_nla.get_global_worker()
    assert global_worker.role == WorkerRole.GLOBALWORKER


def test_get_all_local_workers(mocked_nla):
    local_workers = mocked_nla.get_all_local_workers()
    assert len(local_workers) == 3
    for worker_info in local_workers:
        assert worker_info.role == WorkerRole.LOCALWORKER


def test_get_worker_info(mocked_nla):
    expected_id = "localworker1"
    worker_info = mocked_nla.get_worker_info(expected_id)
    assert worker_info.id == expected_id
    assert worker_info.role == WorkerRole.LOCALWORKER
    assert worker_info.db_port == 50001


def test_empty_initialization():
    worker_registry = WorkerRegistry()
    assert not worker_registry.workers_per_id.values()

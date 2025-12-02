import ipaddress
import logging

import pytest

from exaflow import AttrDict
from exaflow.controller import DeploymentType
from exaflow.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    DataModelRegistry,
)
from exaflow.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    WorkerLandscapeAggregator,
)
from exaflow.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    WorkerRegistry,
)
from exaflow.worker_communication import WorkerInfo
from exaflow.worker_communication import WorkerRole


@pytest.fixture
def mocked_worker_infos():
    return [
        WorkerInfo(
            id="globalworker",
            role=WorkerRole.GLOBALWORKER,
            ip=ipaddress.ip_address("127.0.0.1"),
            port=50000,
        ),
        WorkerInfo(
            id="localworker1",
            role=WorkerRole.LOCALWORKER,
            ip=ipaddress.ip_address("127.0.0.2"),
            port=50001,
        ),
        WorkerInfo(
            id="localworker2",
            role=WorkerRole.LOCALWORKER,
            ip=ipaddress.ip_address("127.0.0.3"),
            port=50002,
        ),
        WorkerInfo(
            id="localworker3",
            role=WorkerRole.LOCALWORKER,
            ip=ipaddress.ip_address("127.0.0.4"),
            port=50003,
        ),
    ]


@pytest.fixture
def mocked_wla(mocked_worker_infos):
    aggregator = WorkerLandscapeAggregator(
        logger=logging.getLogger("worker-registry-tests"),
        update_interval=0,
        tasks_timeout=0,
        deployment_type=DeploymentType.LOCAL,
        localworkers=AttrDict({}),
    )
    worker_registry = WorkerRegistry(workers_info=mocked_worker_infos)
    aggregator._set_new_registries(
        worker_registry=worker_registry, data_model_registry=DataModelRegistry()
    )
    return aggregator


def test_get_workers(mocked_wla):
    workers = mocked_wla.get_workers()
    assert len(workers) == 4
    assert len([w for w in workers if w.role == WorkerRole.LOCALWORKER]) == 3
    assert len([w for w in workers if w.role == WorkerRole.GLOBALWORKER]) == 1


def test_get_global_worker(mocked_wla):
    global_worker = mocked_wla.get_global_worker()
    assert global_worker.id == "globalworker"
    assert global_worker.role == WorkerRole.GLOBALWORKER


def test_get_all_local_workers(mocked_wla):
    local_workers = mocked_wla.get_all_local_workers()
    assert len(local_workers) == 3
    assert all(worker.role == WorkerRole.LOCALWORKER for worker in local_workers)


def test_get_worker_info(mocked_wla):
    worker_info = mocked_wla.get_worker_info("localworker1")
    assert worker_info.role == WorkerRole.LOCALWORKER
    assert worker_info.port == 50001


def test_worker_registry_empty_initialization():
    worker_registry = WorkerRegistry()
    assert not worker_registry.workers_per_id

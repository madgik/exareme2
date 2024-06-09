from typing import Dict
from unittest.mock import Mock

import pytest

from exareme2.controller.services import WorkerLandscapeAggregator
from exareme2.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    DataModelRegistry,
)
from exareme2.utils import AttrDict
from exareme2.worker_communication import CommonDataElement
from exareme2.worker_communication import DataModelMetadata
from tests.standalone_tests.conftest import RABBITMQ_LOCALWORKERTMP_ADDR


@pytest.fixture
def controller_config():
    return {
        "deployment_type": "LOCAL",
        "worker_landscape_aggregator_update_interval": 30,
        "flower_execution_timeout": 30,
        "rabbitmq": {
            "celery_tasks_timeout": 5,
            "celery_run_udf_task_timeout": 10,
        },
        "localworkers": {
            "config_file": None,
        },
    }


@pytest.fixture(scope="function")
def worker_landscape_aggregator(controller_config):
    controller_config = AttrDict(controller_config)
    return WorkerLandscapeAggregator(
        logger=Mock(),
        update_interval=controller_config.worker_landscape_aggregator_update_interval,
        tasks_timeout=controller_config.rabbitmq.celery_tasks_timeout,
        run_udf_task_timeout=controller_config.rabbitmq.celery_run_udf_task_timeout,
        deployment_type=controller_config.deployment_type,
        localworkers=controller_config.localworkers,
    )


def get_parametrization_cases():
    return [
        pytest.param(
            {
                "localworker1": {
                    "data_models_metadata": {
                        "data_model:1": DataModelMetadata(
                            code="data_model:1",
                            version="v1",
                            label="Data Model 1",
                            variables=[
                                CommonDataElement(
                                    is_categorical=True,
                                    code="dataset",
                                    sql_type="text",
                                    label="Dataset",
                                    enumerations={
                                        "dataset1": "DATASET1",
                                        "dataset2": "DATASET2",
                                        "dataset3": "DATASET3",
                                    },
                                    min=None,
                                    max=None,
                                )
                            ],
                            groups=[],
                            longitudinal=False,
                        )
                    },
                    "datasets_per_data_model": {
                        "data_model:1": ["dataset1", "dataset2"]
                    },
                },
                "localworker2": {
                    "data_models_metadata": {
                        "data_model:1": DataModelMetadata(
                            code="data_model:1",
                            version="v1",
                            label="Data Model 1",
                            variables=[
                                CommonDataElement(
                                    is_categorical=True,
                                    code="dataset",
                                    sql_type="text",
                                    label="Dataset",
                                    enumerations={
                                        "dataset1": "DATASET1",
                                        "dataset2": "DATASET2",
                                        "dataset3": "DATASET3",
                                    },
                                    min=None,
                                    max=None,
                                )
                            ],
                            groups=[],
                            longitudinal=False,
                        )
                    },
                    "datasets_per_data_model": {"data_model:1": ["dataset3"]},
                },
            },
            DataModelRegistry(
                data_models_metadata={
                    "data_model:1": DataModelMetadata(
                        code="data_model:1",
                        version="v1",
                        label="Data Model 1",
                        variables=[
                            CommonDataElement(
                                is_categorical=True,
                                code="dataset",
                                sql_type="text",
                                label="Dataset",
                                enumerations={
                                    "dataset1": "DATASET1",
                                    "dataset2": "DATASET2",
                                    "dataset3": "DATASET3",
                                },
                                min=None,
                                max=None,
                            )
                        ],
                        groups=[],
                        longitudinal=False,
                    )
                },
                datasets_locations={
                    "data_model:1": {
                        "dataset1": "localworker1",
                        "dataset2": "localworker1",
                        "dataset3": "localworker2",
                    }
                },
            ),
            id="common_case",
        ),
        pytest.param(
            {
                "localworker1": {
                    "data_models_metadata": {
                        "data_model:1": DataModelMetadata(
                            code="data_model:1",
                            version="v1",
                            label="Data Model 1",
                            variables=[
                                CommonDataElement(
                                    is_categorical=True,
                                    code="dataset",
                                    sql_type="text",
                                    label="Dataset",
                                    enumerations={
                                        "dataset1": "DATASET1",
                                        "dataset2": "DATASET2",
                                    },
                                    min=None,
                                    max=None,
                                )
                            ],
                            groups=[],
                            longitudinal=False,
                        ),
                        "data_model:2": DataModelMetadata(
                            code="data_model:2",
                            version="v1",
                            label="Data Model 2",
                            variables=[
                                CommonDataElement(
                                    is_categorical=True,
                                    code="dataset",
                                    sql_type="text",
                                    label="Dataset",
                                    enumerations={
                                        "dataset3": "DATASET3",
                                        "dataset4": "DATASET4",
                                    },
                                    min=None,
                                    max=None,
                                )
                            ],
                            groups=[],
                            longitudinal=False,
                        ),
                    },
                    "datasets_per_data_model": {
                        "data_model:1": ["dataset1"],
                        "data_model:2": ["dataset3", "dataset4"],
                    },
                },
                "localworker2": {
                    "data_models_metadata": {
                        "data_model:1": DataModelMetadata(
                            code="data_model:1",
                            version="v1",
                            label="Data Model 1",
                            variables=[
                                CommonDataElement(
                                    is_categorical=True,
                                    code="dataset",
                                    sql_type="text",
                                    label="Dataset",
                                    enumerations={
                                        "dataset1": "DATASET1",
                                        "dataset2": "DATASET2",
                                    },
                                    min=None,
                                    max=None,
                                )
                            ],
                            groups=[],
                            longitudinal=False,
                        ),
                        "data_model:2": DataModelMetadata(
                            code="data_model:2",
                            version="v2",
                            label="Data Model 2",
                            variables=[
                                CommonDataElement(
                                    is_categorical=True,
                                    code="dataset",
                                    sql_type="text",
                                    label="Dataset",
                                    enumerations={
                                        "dataset7": "DATASET7",
                                        "dataset8": "DATASET8",
                                    },
                                    min=None,
                                    max=None,
                                )
                            ],
                            groups=[],
                            longitudinal=False,
                        ),
                    },
                    "datasets_per_data_model": {
                        "data_model:1": ["dataset2"],
                        "data_model:2": ["dataset7", "dataset8"],
                    },
                },
            },
            DataModelRegistry(
                data_models_metadata={
                    "data_model:1": DataModelMetadata(
                        code="data_model:1",
                        version="v1",
                        label="Data Model 1",
                        variables=[
                            CommonDataElement(
                                is_categorical=True,
                                code="dataset",
                                sql_type="text",
                                label="Dataset",
                                enumerations={
                                    "dataset1": "DATASET1",
                                    "dataset2": "DATASET2",
                                },
                                min=None,
                                max=None,
                            )
                        ],
                        groups=[],
                        longitudinal=False,
                    )
                },
                datasets_locations={
                    "data_model:1": {
                        "dataset1": "localworker1",
                        "dataset2": "localworker2",
                    }
                },
            ),
            id="incompatible_case",
        ),
        pytest.param(
            {
                "localworker1": {
                    "data_models_metadata": {
                        "data_model:1": DataModelMetadata(
                            code="data_model:1",
                            version="v1",
                            label="Data Model 1",
                            variables=[
                                CommonDataElement(
                                    is_categorical=True,
                                    code="dataset",
                                    sql_type="text",
                                    label="Dataset",
                                    enumerations={
                                        "dataset1": "DATASET1",
                                        "dataset2": "DATASET2",
                                    },
                                    min=None,
                                    max=None,
                                )
                            ],
                            groups=[],
                            longitudinal=False,
                        )
                    },
                    "datasets_per_data_model": {"data_model:1": ["dataset1"]},
                },
                "localworker2": {
                    "data_models_metadata": {
                        "data_model:1": DataModelMetadata(
                            code="data_model:1",
                            version="v1",
                            label="Data Model 1",
                            variables=[
                                CommonDataElement(
                                    is_categorical=True,
                                    code="dataset",
                                    sql_type="text",
                                    label="Dataset",
                                    enumerations={
                                        "dataset1": "DATASET1",
                                        "dataset2": "DATASET2",
                                    },
                                    min=None,
                                    max=None,
                                )
                            ],
                            groups=[],
                            longitudinal=False,
                        )
                    },
                    "datasets_per_data_model": {
                        "data_model:1": ["dataset1", "dataset2"]
                    },
                },
            },
            DataModelRegistry(
                data_models_metadata={
                    "data_model:1": DataModelMetadata(
                        code="data_model:1",
                        version="v1",
                        label="Data Model 1",
                        variables=[
                            CommonDataElement(
                                is_categorical=True,
                                code="dataset",
                                sql_type="text",
                                label="Dataset",
                                enumerations={
                                    "dataset1": "DATASET1",
                                    "dataset2": "DATASET2",
                                },
                                min=None,
                                max=None,
                            )
                        ],
                        groups=[],
                        longitudinal=False,
                    )
                },
                datasets_locations={
                    "data_model:1": {"dataset2": "localworker2"},
                },
            ),
            id="duplicated_datasets_case",
        ),
        pytest.param(
            {
                "localworker1": {
                    "data_models_metadata": {},
                    "datasets_per_data_model": {},
                },
                "localworker2": {
                    "data_models_metadata": {},
                    "datasets_per_data_model": {},
                },
            },
            DataModelRegistry(data_models_metadata={}, datasets_locations={}),
            id="no_data_model_no_datasets",
        ),
        pytest.param(
            {
                "localworker1": {
                    "data_models_metadata": {
                        "data_model:1": DataModelMetadata(
                            code="data_model:1",
                            version="v1",
                            label="Data Model 1",
                            variables=[
                                CommonDataElement(
                                    is_categorical=True,
                                    code="dataset",
                                    sql_type="text",
                                    label="Dataset",
                                    enumerations={
                                        "dataset1": "DATASET1",
                                        "dataset2": "DATASET2",
                                    },
                                    min=None,
                                    max=None,
                                )
                            ],
                            groups=[],
                            longitudinal=False,
                        )
                    },
                    "datasets_per_data_model": {
                        "data_model:1": ["dataset1", "dataset2"]
                    },
                },
                "localworker2": {
                    "data_models_metadata": {
                        "data_model:1": DataModelMetadata(
                            code="data_model:1",
                            version="v1",
                            label="Data Model 1",
                            variables=[
                                CommonDataElement(
                                    is_categorical=True,
                                    code="dataset",
                                    sql_type="text",
                                    label="Dataset",
                                    enumerations={
                                        "dataset1": "DATASET1",
                                        "dataset2": "DATASET2",
                                    },
                                    min=None,
                                    max=None,
                                )
                            ],
                            groups=[],
                            longitudinal=False,
                        )
                    },
                    "datasets_per_data_model": {
                        "data_model:1": ["dataset1", "dataset2"]
                    },
                },
            },
            DataModelRegistry(
                data_models_metadata={
                    "data_model:1": DataModelMetadata(
                        code="data_model:1",
                        version="v1",
                        label="Data Model 1",
                        variables=[
                            CommonDataElement(
                                is_categorical=True,
                                code="dataset",
                                sql_type="text",
                                label="Dataset",
                                enumerations={
                                    "dataset1": "DATASET1",
                                    "dataset2": "DATASET2",
                                },
                                min=None,
                                max=None,
                            )
                        ],
                        groups=[],
                        longitudinal=False,
                    )
                },
                datasets_locations={},
            ),
            id="identical_data_models_and_datasets_case",
        ),
    ]


@pytest.mark.parametrize("workers_data, expected", get_parametrization_cases())
def test_data_model_registry(
    workers_data: dict, expected: DataModelRegistry, worker_landscape_aggregator
):
    aggregate_data_models_metadata = {}
    incompatible_data_models = {}

    for worker_id, data in workers_data.items():
        models_metadata = data["data_models_metadata"]
        worker_landscape_aggregator._process_data_models_metadata(
            models_metadata,
            aggregate_data_models_metadata,
            incompatible_data_models,
            worker_id,
        )

    datasets_locations = {}
    duplicated_datasets = {}

    for worker_id, data in workers_data.items():
        datasets_per_data_model = data["datasets_per_data_model"]
        worker_landscape_aggregator._process_datasets_location(
            datasets_per_data_model, datasets_locations, duplicated_datasets, worker_id
        )

    worker_landscape_aggregator._cleanup_incompatible_data_models(
        aggregate_data_models_metadata, datasets_locations, incompatible_data_models
    )

    worker_landscape_aggregator._log_and_cleanup_duplicated_datasets(
        datasets_locations, duplicated_datasets
    )

    result = DataModelRegistry(
        data_models_metadata=aggregate_data_models_metadata,
        datasets_locations=datasets_locations,
    )

    assert result.data_models_metadata == expected.data_models_metadata
    assert result.datasets_locations == expected.datasets_locations


@pytest.mark.slow
def test_get_workers_info_properly_handles_errors(worker_landscape_aggregator):
    workers_info = worker_landscape_aggregator._get_workers_info(
        [RABBITMQ_LOCALWORKERTMP_ADDR]
    )
    assert not workers_info


@pytest.mark.slow
def test__get_worker_data_model_metadata_and_datasets_properly_handles_errors(
    worker_landscape_aggregator,
):
    (
        data_model_metadata,
        datasets,
    ) = worker_landscape_aggregator._get_worker_data_model_metadata_and_datasets(
        RABBITMQ_LOCALWORKERTMP_ADDR
    )
    assert not data_model_metadata
    assert not datasets

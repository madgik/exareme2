import pytest

from exareme2 import AttrDict
from exareme2.controller import logger as ctrl_logger
from exareme2.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    DataModelMetadata,
)
from exareme2.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    DataModelRegistry,
)
from exareme2.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    DataModelsAttributes,
)
from exareme2.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    DataModelsCDES,
)
from exareme2.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    DataModelsMetadata,
)
from exareme2.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    DataModelsMetadataPerWorker,
)
from exareme2.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    DatasetsLabels,
)
from exareme2.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    DatasetsLocations,
)
from exareme2.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    WorkerLandscapeAggregator,
)
from exareme2.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    _crunch_data_model_registry_data,
)
from exareme2.worker_communication import CommonDataElement
from exareme2.worker_communication import CommonDataElements
from exareme2.worker_communication import DataModelAttributes
from tests.standalone_tests.conftest import RABBITMQ_LOCALWORKERTMP_ADDR


@pytest.fixture
def controller_config():
    controller_config = {
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
    return controller_config


@pytest.fixture(scope="function")
def worker_landscape_aggregator(
    controller_config,
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
    return worker_landscape_aggregator


def get_parametrization_cases():
    return [
        pytest.param(
            DataModelsMetadataPerWorker(
                data_models_metadata_per_worker={
                    "localworker1": DataModelsMetadata(
                        data_models_metadata={
                            "data_model:1": DataModelMetadata(
                                datasets_labels=DatasetsLabels(
                                    datasets_labels={
                                        "dataset1": "DATASET1",
                                        "dataset2": "DATASET2",
                                    }
                                ),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset1": "DATASET1",
                                                "dataset2": "DATASET2",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=["tag1"],
                                    properties={"key1": "value1"},
                                ),
                            ),
                            "data_model:2": DataModelMetadata(
                                datasets_labels=DatasetsLabels(
                                    datasets_labels={
                                        "dataset1": "DATASET1",
                                    }
                                ),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset1": "DATASET1",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=["tag1"],
                                    properties={"key1": "value1"},
                                ),
                            ),
                        }
                    ),
                    "localworker2": DataModelsMetadata(
                        data_models_metadata={
                            "data_model:1": DataModelMetadata(
                                datasets_labels=DatasetsLabels(
                                    datasets_labels={
                                        "dataset3": "DATASET3",
                                        "dataset4": "DATASET4",
                                    }
                                ),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset3": "DATASET3",
                                                "dataset4": "DATASET4",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=["tag2"],
                                    properties={"key2": "value2"},
                                ),
                            ),
                        }
                    ),
                    "localworker3": DataModelsMetadata(
                        data_models_metadata={
                            "data_model:1": DataModelMetadata(
                                datasets_labels=DatasetsLabels(
                                    datasets_labels={
                                        "dataset5": "DATASET5",
                                        "dataset6": "DATASET6",
                                    }
                                ),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset5": "DATASET5",
                                                "dataset6": "DATASET6",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=["tag3"],
                                    properties={"key3": "value3"},
                                ),
                            ),
                            "data_model:2": DataModelMetadata(
                                datasets_labels=DatasetsLabels(
                                    datasets_labels={
                                        "dataset2": "DATASET2",
                                    }
                                ),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset2": "DATASET2",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=["tag2"],
                                    properties={"key2": "value2"},
                                ),
                            ),
                        }
                    ),
                }
            ),
            DataModelRegistry(
                data_models_attributes=DataModelsAttributes(
                    data_models_attributes={
                        "data_model:1": DataModelAttributes(
                            tags=["tag1", "tag2", "tag3"],
                            properties={
                                "key1": ["value1"],
                                "key2": ["value2"],
                                "key3": ["value3"],
                            },
                        ),
                        "data_model:2": DataModelAttributes(
                            tags=["tag1", "tag2"],
                            properties={
                                "key1": ["value1"],
                                "key2": ["value2"],
                            },
                        ),
                    }
                ),
                data_models_cdes=DataModelsCDES(
                    data_models_cdes={
                        "data_model:1": CommonDataElements(
                            values={
                                "dataset": CommonDataElement(
                                    code="dataset",
                                    label="Dataset",
                                    sql_type="text",
                                    is_categorical=True,
                                    enumerations={
                                        "dataset1": "DATASET1",
                                        "dataset2": "DATASET2",
                                        "dataset3": "DATASET3",
                                        "dataset4": "DATASET4",
                                        "dataset5": "DATASET5",
                                        "dataset6": "DATASET6",
                                    },
                                    min=None,
                                    max=None,
                                )
                            }
                        ),
                        "data_model:2": CommonDataElements(
                            values={
                                "dataset": CommonDataElement(
                                    code="dataset",
                                    label="Dataset",
                                    sql_type="text",
                                    is_categorical=True,
                                    enumerations={
                                        "dataset1": "DATASET1",
                                        "dataset2": "DATASET2",
                                    },
                                    min=None,
                                    max=None,
                                )
                            }
                        ),
                    }
                ),
                datasets_locations=DatasetsLocations(
                    datasets_locations={
                        "data_model:1": {
                            "dataset1": "localworker1",
                            "dataset2": "localworker1",
                            "dataset3": "localworker2",
                            "dataset4": "localworker2",
                            "dataset5": "localworker3",
                            "dataset6": "localworker3",
                        },
                        "data_model:2": {
                            "dataset1": "localworker1",
                            "dataset2": "localworker3",
                        },
                    }
                ),
            ),
            id="common_case",
        ),
        pytest.param(
            DataModelsMetadataPerWorker(
                data_models_metadata_per_worker={
                    "localworker1": DataModelsMetadata(
                        data_models_metadata={
                            "data_model:1": DataModelMetadata(
                                datasets_labels=DatasetsLabels(
                                    datasets_labels={
                                        "dataset1": "DATASET1",
                                        "dataset2": "DATASET2",
                                    }
                                ),
                                cdes=None,
                                attributes=DataModelAttributes(
                                    tags=[],
                                    properties={},
                                ),
                            ),
                            "data_model:2": DataModelMetadata(
                                datasets_labels=DatasetsLabels(
                                    datasets_labels={
                                        "dataset1": "DATASET1",
                                    }
                                ),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset1": "DATASET1",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=[],
                                    properties={},
                                ),
                            ),
                        }
                    ),
                    "localworker2": DataModelsMetadata(
                        data_models_metadata={
                            "data_model:1": DataModelMetadata(
                                datasets_labels=DatasetsLabels(
                                    datasets_labels={
                                        "dataset3": "DATASET3",
                                        "dataset4": "DATASET4",
                                    }
                                ),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset3": "DATASET3",
                                                "dataset4": "DATASET4",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=[],
                                    properties={},
                                ),
                            ),
                        }
                    ),
                    "localworker3": DataModelsMetadata(
                        data_models_metadata={
                            "data_model:1": DataModelMetadata(
                                datasets_labels=DatasetsLabels(
                                    datasets_labels={
                                        "dataset5": "DATASET5",
                                        "dataset6": "DATASET6",
                                    }
                                ),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset5": "DATASET5",
                                                "dataset6": "DATASET6",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=[],
                                    properties={},
                                ),
                            ),
                            "data_model:2": DataModelMetadata(
                                datasets_labels=DatasetsLabels(
                                    datasets_labels={
                                        "dataset2": "DATASET2",
                                    }
                                ),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset2": "DATASET2",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=[],
                                    properties={},
                                ),
                            ),
                        }
                    ),
                }
            ),
            DataModelRegistry(
                data_models_attributes=DataModelsAttributes(
                    data_models_attributes={
                        "data_model:2": DataModelAttributes(tags=[], properties={}),
                        "data_model:1": DataModelAttributes(tags=[], properties={}),
                    }
                ),
                data_models_cdes=DataModelsCDES(
                    data_models_cdes={
                        "data_model:1": CommonDataElements(
                            values={
                                "dataset": CommonDataElement(
                                    code="dataset",
                                    label="Dataset",
                                    sql_type="text",
                                    is_categorical=True,
                                    enumerations={
                                        "dataset3": "DATASET3",
                                        "dataset4": "DATASET4",
                                        "dataset5": "DATASET5",
                                        "dataset6": "DATASET6",
                                    },
                                    min=None,
                                    max=None,
                                )
                            }
                        ),
                        "data_model:2": CommonDataElements(
                            values={
                                "dataset": CommonDataElement(
                                    code="dataset",
                                    label="Dataset",
                                    sql_type="text",
                                    is_categorical=True,
                                    enumerations={
                                        "dataset1": "DATASET1",
                                        "dataset2": "DATASET2",
                                    },
                                    min=None,
                                    max=None,
                                )
                            }
                        ),
                    }
                ),
                datasets_locations=DatasetsLocations(
                    datasets_locations={
                        "data_model:1": {
                            "dataset3": "localworker2",
                            "dataset4": "localworker2",
                            "dataset5": "localworker3",
                            "dataset6": "localworker3",
                        },
                        "data_model:2": {
                            "dataset1": "localworker1",
                            "dataset2": "localworker3",
                        },
                    }
                ),
            ),
            id="none_cdes_on_data_model_1",
        ),
        pytest.param(
            DataModelsMetadataPerWorker(
                data_models_metadata_per_worker={
                    "localworker1": DataModelsMetadata(
                        data_models_metadata={
                            "data_model:1": DataModelMetadata(
                                datasets_labels=DatasetsLabels(
                                    datasets_labels={
                                        "dataset1": "DATASET1",
                                        "dataset2": "DATASET2",
                                    }
                                ),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset1": "DATASET1",
                                                "dataset2": "DATASET2",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                        "age_value": CommonDataElement(
                                            code="age_value",
                                            label="Age",
                                            sql_type="int",
                                            is_categorical=False,
                                            enumerations=None,
                                            min=0.0,
                                            max=130.0,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=[],
                                    properties={},
                                ),
                            ),
                            "data_model:2": DataModelMetadata(
                                datasets_labels=DatasetsLabels(
                                    datasets_labels={
                                        "dataset1": "DATASET1",
                                    }
                                ),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset1": "DATASET1",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=[],
                                    properties={},
                                ),
                            ),
                        }
                    ),
                    "localworker2": DataModelsMetadata(
                        data_models_metadata={
                            "data_model:1": DataModelMetadata(
                                datasets_labels=DatasetsLabels(
                                    datasets_labels={
                                        "dataset3": "DATASET3",
                                        "dataset4": "DATASET4",
                                    }
                                ),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset3": "DATASET3",
                                                "dataset4": "DATASET4",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                        "age_value": CommonDataElement(
                                            code="age_value",
                                            label="Age",
                                            sql_type="int",
                                            is_categorical=False,
                                            enumerations=None,
                                            min=1.0,
                                            max=130.0,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=[],
                                    properties={},
                                ),
                            ),
                        }
                    ),
                    "localworker3": DataModelsMetadata(
                        data_models_metadata={
                            "data_model:1": DataModelMetadata(
                                datasets_labels=DatasetsLabels(
                                    datasets_labels={
                                        "dataset5": "DATASET5",
                                        "dataset6": "DATASET6",
                                    }
                                ),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset5": "DATASET5",
                                                "dataset6": "DATASET6",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                        "age_value": CommonDataElement(
                                            code="age_value",
                                            label="Age",
                                            sql_type="int",
                                            is_categorical=False,
                                            enumerations=None,
                                            min=0.0,
                                            max=130.0,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=[],
                                    properties={},
                                ),
                            ),
                            "data_model:2": DataModelMetadata(
                                datasets_labels=DatasetsLabels(
                                    datasets_labels={
                                        "dataset2": "DATASET2",
                                    }
                                ),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset2": "DATASET2",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=[],
                                    properties={},
                                ),
                            ),
                        }
                    ),
                }
            ),
            DataModelRegistry(
                data_models_attributes=DataModelsAttributes(
                    data_models_attributes={
                        "data_model:2": DataModelAttributes(tags=[], properties={})
                    }
                ),
                data_models_cdes=DataModelsCDES(
                    data_models_cdes={
                        "data_model:2": CommonDataElements(
                            values={
                                "dataset": CommonDataElement(
                                    code="dataset",
                                    label="Dataset",
                                    sql_type="text",
                                    is_categorical=True,
                                    enumerations={
                                        "dataset1": "DATASET1",
                                        "dataset2": "DATASET2",
                                    },
                                    min=None,
                                    max=None,
                                )
                            }
                        ),
                    }
                ),
                datasets_locations=DatasetsLocations(
                    datasets_locations={
                        "data_model:2": {
                            "dataset1": "localworker1",
                            "dataset2": "localworker3",
                        },
                    }
                ),
            ),
            id="incompatible_cdes_on_data_model1_on_worker1_and_worker3",
        ),
        pytest.param(
            DataModelsMetadataPerWorker(
                data_models_metadata_per_worker={
                    "localworker1": DataModelsMetadata(
                        data_models_metadata={
                            "data_model:1": DataModelMetadata(
                                datasets_labels=DatasetsLabels(datasets_labels={}),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset1": "DATASET1",
                                                "dataset2": "DATASET2",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=[],
                                    properties={},
                                ),
                            ),
                            "data_model:2": DataModelMetadata(
                                datasets_labels=DatasetsLabels(datasets_labels={}),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset1": "DATASET1",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=[],
                                    properties={},
                                ),
                            ),
                        }
                    ),
                    "localworker2": DataModelsMetadata(
                        data_models_metadata={
                            "data_model:1": DataModelMetadata(
                                datasets_labels=DatasetsLabels(datasets_labels={}),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset3": "DATASET3",
                                                "dataset4": "DATASET4",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=[],
                                    properties={},
                                ),
                            ),
                        }
                    ),
                    "localworker3": DataModelsMetadata(
                        data_models_metadata={
                            "data_model:1": DataModelMetadata(
                                datasets_labels=DatasetsLabels(datasets_labels={}),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset5": "DATASET5",
                                                "dataset6": "DATASET6",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=[],
                                    properties={},
                                ),
                            ),
                            "data_model:2": DataModelMetadata(
                                datasets_labels=DatasetsLabels(datasets_labels={}),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset2": "DATASET2",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=[],
                                    properties={},
                                ),
                            ),
                        }
                    ),
                }
            ),
            DataModelRegistry(
                data_models_attributes=DataModelsAttributes(
                    data_models_attributes={
                        "data_model:1": DataModelAttributes(tags=[], properties={}),
                        "data_model:2": DataModelAttributes(tags=[], properties={}),
                    }
                ),
                data_models_cdes=DataModelsCDES(
                    data_models_cdes={
                        "data_model:1": CommonDataElements(
                            values={
                                "dataset": CommonDataElement(
                                    code="dataset",
                                    label="Dataset",
                                    sql_type="text",
                                    is_categorical=True,
                                    enumerations={},
                                    min=None,
                                    max=None,
                                )
                            }
                        ),
                        "data_model:2": CommonDataElements(
                            values={
                                "dataset": CommonDataElement(
                                    code="dataset",
                                    label="Dataset",
                                    sql_type="text",
                                    is_categorical=True,
                                    enumerations={},
                                    min=None,
                                    max=None,
                                )
                            }
                        ),
                    }
                ),
                datasets_locations=DatasetsLocations(
                    datasets_locations={
                        "data_model:1": {},
                        "data_model:2": {},
                    }
                ),
            ),
            id="no_data_model_or_dataset_case",
        ),
        pytest.param(
            DataModelsMetadataPerWorker(
                data_models_metadata_per_worker={
                    "localworker1": DataModelsMetadata(data_models_metadata={}),
                    "localworker2": DataModelsMetadata(data_models_metadata={}),
                }
            ),
            DataModelRegistry(
                data_models_attributes=DataModelsAttributes(data_models_attributes={}),
                data_models_cdes=DataModelsCDES(data_models_cdes={}),
                datasets_locations=DatasetsLocations(datasets_locations={}),
            ),
            id="no_data_model_or_dataset_case",
        ),
        pytest.param(
            DataModelsMetadataPerWorker(
                data_models_metadata_per_worker={
                    "localworker1": DataModelsMetadata(
                        data_models_metadata={
                            "data_model:1": DataModelMetadata(
                                datasets_labels=DatasetsLabels(
                                    datasets_labels={
                                        "dataset1": "DATASET1",
                                        "dataset2": "DATASET2",
                                    }
                                ),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset1": "DATASET1",
                                                "dataset2": "DATASET2",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=[],
                                    properties={},
                                ),
                            ),
                            "data_model:2": DataModelMetadata(
                                datasets_labels=DatasetsLabels(
                                    datasets_labels={
                                        "dataset1": "DATASET1",
                                    }
                                ),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset1": "DATASET1",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=[],
                                    properties={},
                                ),
                            ),
                        }
                    ),
                    "localworker2": DataModelsMetadata(
                        data_models_metadata={
                            "data_model:1": DataModelMetadata(
                                datasets_labels=DatasetsLabels(
                                    datasets_labels={
                                        "dataset1": "DATASET1",
                                        "dataset2": "DATASET2",
                                    }
                                ),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset1": "DATASET1",
                                                "dataset2": "DATASET2",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=[],
                                    properties={},
                                ),
                            ),
                            "data_model:2": DataModelMetadata(
                                datasets_labels=DatasetsLabels(
                                    datasets_labels={
                                        "dataset1": "DATASET1",
                                    }
                                ),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset1": "DATASET1",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=[],
                                    properties={},
                                ),
                            ),
                        }
                    ),
                    "localworker3": DataModelsMetadata(
                        data_models_metadata={
                            "data_model:1": DataModelMetadata(
                                datasets_labels=DatasetsLabels(
                                    datasets_labels={
                                        "dataset1": "DATASET1",
                                        "dataset2": "DATASET2",
                                    }
                                ),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset1": "DATASET1",
                                                "dataset2": "DATASET2",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=[],
                                    properties={},
                                ),
                            ),
                            "data_model:2": DataModelMetadata(
                                datasets_labels=DatasetsLabels(
                                    datasets_labels={
                                        "dataset1": "DATASET1",
                                    }
                                ),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset1": "DATASET1",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=[],
                                    properties={},
                                ),
                            ),
                        }
                    ),
                    "localworker4": DataModelsMetadata(
                        data_models_metadata={
                            "data_model:1": DataModelMetadata(
                                datasets_labels=DatasetsLabels(
                                    datasets_labels={
                                        "dataset1": "DATASET1",
                                        "dataset2": "DATASET2",
                                    }
                                ),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset1": "DATASET1",
                                                "dataset2": "DATASET2",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=[],
                                    properties={},
                                ),
                            ),
                            "data_model:2": DataModelMetadata(
                                datasets_labels=DatasetsLabels(
                                    datasets_labels={
                                        "dataset1": "DATASET1",
                                    }
                                ),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset1": "DATASET1",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=[],
                                    properties={},
                                ),
                            ),
                        }
                    ),
                }
            ),
            DataModelRegistry(
                data_models_attributes=DataModelsAttributes(
                    data_models_attributes={
                        "data_model:1": DataModelAttributes(tags=[], properties={}),
                        "data_model:2": DataModelAttributes(tags=[], properties={}),
                    }
                ),
                data_models_cdes=DataModelsCDES(
                    data_models_cdes={
                        "data_model:1": CommonDataElements(
                            values={
                                "dataset": CommonDataElement(
                                    code="dataset",
                                    label="Dataset",
                                    sql_type="text",
                                    is_categorical=True,
                                    enumerations={},
                                    min=None,
                                    max=None,
                                )
                            }
                        ),
                        "data_model:2": CommonDataElements(
                            values={
                                "dataset": CommonDataElement(
                                    code="dataset",
                                    label="Dataset",
                                    sql_type="text",
                                    is_categorical=True,
                                    enumerations={},
                                    min=None,
                                    max=None,
                                )
                            }
                        ),
                    }
                ),
                datasets_locations=DatasetsLocations(
                    datasets_locations={"data_model:1": {}, "data_model:2": {}}
                ),
            ),
            id="same_data_models_and_datasets_on_all_workers",
        ),
        pytest.param(
            DataModelsMetadataPerWorker(
                data_models_metadata_per_worker={
                    "localworker1": DataModelsMetadata(
                        data_models_metadata={
                            "data_model:1": DataModelMetadata(
                                datasets_labels=DatasetsLabels(
                                    datasets_labels={
                                        "dataset1": "DATASET1",
                                        "dataset2": "DATASET2",
                                    }
                                ),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset1": "DATASET1",
                                                "dataset2": "DATASET2",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=[],
                                    properties={},
                                ),
                            ),
                            "data_model:2": DataModelMetadata(
                                datasets_labels=DatasetsLabels(
                                    datasets_labels={
                                        "dataset1": "DATASET1",
                                    }
                                ),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset1": "DATASET1",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=[],
                                    properties={},
                                ),
                            ),
                        }
                    ),
                    "localworker2": DataModelsMetadata(
                        data_models_metadata={
                            "data_model:1": DataModelMetadata(
                                datasets_labels=DatasetsLabels(
                                    datasets_labels={
                                        "dataset3": "DATASET3",
                                        "dataset4": "DATASET4",
                                    }
                                ),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset3": "DATASET3",
                                                "dataset4": "DATASET4",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=[],
                                    properties={},
                                ),
                            ),
                        }
                    ),
                    "localworker3": DataModelsMetadata(
                        data_models_metadata={
                            "data_model:1": DataModelMetadata(
                                datasets_labels=DatasetsLabels(
                                    datasets_labels={
                                        "dataset5": "DATASET5",
                                        "dataset6": "DATASET6",
                                    }
                                ),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset5": "DATASET5",
                                                "dataset6": "DATASET6",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=[],
                                    properties={},
                                ),
                            ),
                            "data_model:2": DataModelMetadata(
                                datasets_labels=DatasetsLabels(
                                    datasets_labels={
                                        "dataset1": "DATASET1",
                                    }
                                ),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset1": "DATASET1",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=[],
                                    properties={},
                                ),
                            ),
                        }
                    ),
                }
            ),
            DataModelRegistry(
                data_models_attributes=DataModelsAttributes(
                    data_models_attributes={
                        "data_model:1": DataModelAttributes(tags=[], properties={}),
                        "data_model:2": DataModelAttributes(tags=[], properties={}),
                    }
                ),
                data_models_cdes=DataModelsCDES(
                    data_models_cdes={
                        "data_model:1": CommonDataElements(
                            values={
                                "dataset": CommonDataElement(
                                    code="dataset",
                                    label="Dataset",
                                    sql_type="text",
                                    is_categorical=True,
                                    enumerations={
                                        "dataset1": "DATASET1",
                                        "dataset2": "DATASET2",
                                        "dataset3": "DATASET3",
                                        "dataset4": "DATASET4",
                                        "dataset5": "DATASET5",
                                        "dataset6": "DATASET6",
                                    },
                                    min=None,
                                    max=None,
                                )
                            }
                        ),
                        "data_model:2": CommonDataElements(
                            values={
                                "dataset": CommonDataElement(
                                    code="dataset",
                                    label="Dataset",
                                    sql_type="text",
                                    is_categorical=True,
                                    enumerations={},
                                    min=None,
                                    max=None,
                                )
                            }
                        ),
                    }
                ),
                datasets_locations=DatasetsLocations(
                    datasets_locations={
                        "data_model:1": {
                            "dataset1": "localworker1",
                            "dataset2": "localworker1",
                            "dataset3": "localworker2",
                            "dataset4": "localworker2",
                            "dataset5": "localworker3",
                            "dataset6": "localworker3",
                        },
                        "data_model:2": {},
                    }
                ),
            ),
            id="duplicated_dataset1_on_data_model2",
        ),
        pytest.param(
            DataModelsMetadataPerWorker(
                data_models_metadata_per_worker={
                    "localworker1": DataModelsMetadata(
                        data_models_metadata={
                            "data_model:1": DataModelMetadata(
                                datasets_labels=DatasetsLabels(
                                    datasets_labels={
                                        "dataset1": "DATASET1",
                                        "dataset2": "DATASET2",
                                    }
                                ),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset1": "DATASET1",
                                                "dataset2": "DATASET2",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                        "age_value": CommonDataElement(
                                            code="age_value",
                                            label="Age",
                                            sql_type="int",
                                            is_categorical=False,
                                            enumerations=None,
                                            min=0.0,
                                            max=130.0,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=[],
                                    properties={},
                                ),
                            ),
                            "data_model:2": DataModelMetadata(
                                datasets_labels=DatasetsLabels(
                                    datasets_labels={
                                        "dataset1": "DATASET1",
                                    }
                                ),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset1": "DATASET1",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=[],
                                    properties={},
                                ),
                            ),
                        }
                    ),
                    "localworker2": DataModelsMetadata(
                        data_models_metadata={
                            "data_model:1": DataModelMetadata(
                                datasets_labels=DatasetsLabels(
                                    datasets_labels={
                                        "dataset3": "DATASET3",
                                        "dataset4": "DATASET4",
                                    }
                                ),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset3": "DATASET3",
                                                "dataset4": "DATASET4",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=[],
                                    properties={},
                                ),
                            ),
                        }
                    ),
                    "localworker3": DataModelsMetadata(
                        data_models_metadata={
                            "data_model:1": DataModelMetadata(
                                datasets_labels=DatasetsLabels(
                                    datasets_labels={
                                        "dataset5": "DATASET5",
                                        "dataset6": "DATASET6",
                                    }
                                ),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset5": "DATASET5",
                                                "dataset6": "DATASET6",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=[],
                                    properties={},
                                ),
                            ),
                            "data_model:2": DataModelMetadata(
                                datasets_labels=DatasetsLabels(
                                    datasets_labels={
                                        "dataset2": "DATASET2",
                                    }
                                ),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset2": "DATASET2",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=[],
                                    properties={},
                                ),
                            ),
                        }
                    ),
                }
            ),
            DataModelRegistry(
                data_models_attributes=DataModelsAttributes(
                    data_models_attributes={
                        "data_model:2": DataModelAttributes(tags=[], properties={})
                    }
                ),
                data_models_cdes=DataModelsCDES(
                    data_models_cdes={
                        "data_model:2": CommonDataElements(
                            values={
                                "dataset": CommonDataElement(
                                    code="dataset",
                                    label="Dataset",
                                    sql_type="text",
                                    is_categorical=True,
                                    enumerations={
                                        "dataset1": "DATASET1",
                                        "dataset2": "DATASET2",
                                    },
                                    min=None,
                                    max=None,
                                )
                            }
                        ),
                    }
                ),
                datasets_locations=DatasetsLocations(
                    datasets_locations={
                        "data_model:2": {
                            "dataset1": "localworker1",
                            "dataset2": "localworker3",
                        },
                    }
                ),
            ),
            id="incompatible_cdes_on_data_model1",
        ),
        pytest.param(
            DataModelsMetadataPerWorker(
                data_models_metadata_per_worker={
                    "localworker1": DataModelsMetadata(
                        data_models_metadata={
                            "data_model:1": DataModelMetadata(
                                datasets_labels=DatasetsLabels(
                                    datasets_labels={
                                        "dataset1": "DATASET1",
                                        "dataset2": "DATASET2",
                                    }
                                ),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset1": "DATASET1",
                                                "dataset2": "DATASET2",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                        "age_value": CommonDataElement(
                                            code="age_value",
                                            label="Age",
                                            sql_type="int",
                                            is_categorical=False,
                                            enumerations=None,
                                            min=0.0,
                                            max=130.0,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=[],
                                    properties={},
                                ),
                            ),
                            "data_model:2": DataModelMetadata(
                                datasets_labels=DatasetsLabels(
                                    datasets_labels={
                                        "dataset1": "DATASET1",
                                    }
                                ),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset1": "DATASET1",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=[],
                                    properties={},
                                ),
                            ),
                        }
                    ),
                    "localworker2": DataModelsMetadata(
                        data_models_metadata={
                            "data_model:1": DataModelMetadata(
                                datasets_labels=DatasetsLabels(
                                    datasets_labels={
                                        "dataset3": "DATASET3",
                                        "dataset4": "DATASET4",
                                    }
                                ),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset3": "DATASET3",
                                                "dataset4": "DATASET4",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=[],
                                    properties={},
                                ),
                            ),
                        }
                    ),
                    "localworker3": DataModelsMetadata(
                        data_models_metadata={
                            "data_model:1": DataModelMetadata(
                                datasets_labels=DatasetsLabels(
                                    datasets_labels={
                                        "dataset5": "DATASET5",
                                        "dataset6": "DATASET6",
                                    }
                                ),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset5": "DATASET5",
                                                "dataset6": "DATASET6",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=[],
                                    properties={},
                                ),
                            ),
                            "data_model:2": DataModelMetadata(
                                datasets_labels=DatasetsLabels(
                                    datasets_labels={
                                        "dataset2": "DATASET2",
                                    }
                                ),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset2": "DATASET2",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=[],
                                    properties={},
                                ),
                            ),
                        }
                    ),
                }
            ),
            DataModelRegistry(
                data_models_attributes=DataModelsAttributes(
                    data_models_attributes={
                        "data_model:2": DataModelAttributes(tags=[], properties={})
                    }
                ),
                data_models_cdes=DataModelsCDES(
                    data_models_cdes={
                        "data_model:2": CommonDataElements(
                            values={
                                "dataset": CommonDataElement(
                                    code="dataset",
                                    label="Dataset",
                                    sql_type="text",
                                    is_categorical=True,
                                    enumerations={
                                        "dataset1": "DATASET1",
                                        "dataset2": "DATASET2",
                                    },
                                    min=None,
                                    max=None,
                                )
                            }
                        ),
                    }
                ),
                datasets_locations=DatasetsLocations(
                    datasets_locations={
                        "data_model:2": {
                            "dataset1": "localworker1",
                            "dataset2": "localworker3",
                        },
                    }
                ),
            ),
            id="incompatible_cdes_on_data_model1",
        ),
        pytest.param(
            DataModelsMetadataPerWorker(
                data_models_metadata_per_worker={
                    "localworker1": DataModelsMetadata(
                        data_models_metadata={
                            "data_model:1": DataModelMetadata(
                                datasets_labels=DatasetsLabels(
                                    datasets_labels={
                                        "dataset1": "DATASET1",
                                        "dataset2": "DATASET2",
                                    }
                                ),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset1": "DATASET1",
                                                "dataset2": "DATASET2",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=["tag1", "common-tag"],
                                    properties={
                                        "key1": "value1",
                                        "common-key": "common-value",
                                    },
                                ),
                            ),
                            "data_model:2": DataModelMetadata(
                                datasets_labels=DatasetsLabels(
                                    datasets_labels={
                                        "dataset1": "DATASET1",
                                    }
                                ),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset1": "DATASET1",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=["tag1"],
                                    properties={"key1": "value1"},
                                ),
                            ),
                        }
                    ),
                    "localworker2": DataModelsMetadata(
                        data_models_metadata={
                            "data_model:1": DataModelMetadata(
                                datasets_labels=DatasetsLabels(
                                    datasets_labels={
                                        "dataset3": "DATASET3",
                                        "dataset4": "DATASET4",
                                    }
                                ),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset3": "DATASET3",
                                                "dataset4": "DATASET4",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=["tag2", "common-tag"],
                                    properties={
                                        "key2": "value2",
                                        "common-key": "common-value",
                                    },
                                ),
                            ),
                        }
                    ),
                    "localworker3": DataModelsMetadata(
                        data_models_metadata={
                            "data_model:1": DataModelMetadata(
                                datasets_labels=DatasetsLabels(
                                    datasets_labels={
                                        "dataset5": "DATASET5",
                                        "dataset6": "DATASET6",
                                    }
                                ),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset5": "DATASET5",
                                                "dataset6": "DATASET6",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=["tag3"],
                                    properties={"key3": "value3"},
                                ),
                            ),
                            "data_model:2": DataModelMetadata(
                                datasets_labels=DatasetsLabels(
                                    datasets_labels={
                                        "dataset2": "DATASET2",
                                    }
                                ),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset2": "DATASET2",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=["tag2"],
                                    properties={"key2": "value2"},
                                ),
                            ),
                        }
                    ),
                }
            ),
            DataModelRegistry(
                data_models_attributes=DataModelsAttributes(
                    data_models_attributes={
                        "data_model:1": DataModelAttributes(
                            tags=["tag1", "common-tag", "tag2", "tag3"],
                            properties={
                                "key1": ["value1"],
                                "common-key": ["common-value"],
                                "key2": ["value2"],
                                "key3": ["value3"],
                            },
                        ),
                        "data_model:2": DataModelAttributes(
                            tags=["tag1", "tag2"],
                            properties={
                                "key1": ["value1"],
                                "key2": ["value2"],
                            },
                        ),
                    }
                ),
                data_models_cdes=DataModelsCDES(
                    data_models_cdes={
                        "data_model:1": CommonDataElements(
                            values={
                                "dataset": CommonDataElement(
                                    code="dataset",
                                    label="Dataset",
                                    sql_type="text",
                                    is_categorical=True,
                                    enumerations={
                                        "dataset1": "DATASET1",
                                        "dataset2": "DATASET2",
                                        "dataset3": "DATASET3",
                                        "dataset4": "DATASET4",
                                        "dataset5": "DATASET5",
                                        "dataset6": "DATASET6",
                                    },
                                    min=None,
                                    max=None,
                                )
                            }
                        ),
                        "data_model:2": CommonDataElements(
                            values={
                                "dataset": CommonDataElement(
                                    code="dataset",
                                    label="Dataset",
                                    sql_type="text",
                                    is_categorical=True,
                                    enumerations={
                                        "dataset1": "DATASET1",
                                        "dataset2": "DATASET2",
                                    },
                                    min=None,
                                    max=None,
                                )
                            }
                        ),
                    }
                ),
                datasets_locations=DatasetsLocations(
                    datasets_locations={
                        "data_model:1": {
                            "dataset1": "localworker1",
                            "dataset2": "localworker1",
                            "dataset3": "localworker2",
                            "dataset4": "localworker2",
                            "dataset5": "localworker3",
                            "dataset6": "localworker3",
                        },
                        "data_model:2": {
                            "dataset1": "localworker1",
                            "dataset2": "localworker3",
                        },
                    }
                ),
            ),
            id="duplicated_tags_and_properties",
        ),
        pytest.param(
            DataModelsMetadataPerWorker(
                data_models_metadata_per_worker={
                    "localworker1": DataModelsMetadata(
                        data_models_metadata={
                            "data_model:1": DataModelMetadata(
                                datasets_labels=DatasetsLabels(
                                    datasets_labels={
                                        "dataset1": "DATASET1",
                                        "dataset2": "DATASET2",
                                    }
                                ),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset1": "DATASET1",
                                                "dataset2": "DATASET2",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=["tag1"],
                                    properties={
                                        "key1": "value1",
                                        "common-key": "different-value1",
                                    },
                                ),
                            ),
                            "data_model:2": DataModelMetadata(
                                datasets_labels=DatasetsLabels(
                                    datasets_labels={
                                        "dataset1": "DATASET1",
                                    }
                                ),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset1": "DATASET1",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=["tag1"],
                                    properties={"key1": "value1"},
                                ),
                            ),
                        }
                    ),
                    "localworker2": DataModelsMetadata(
                        data_models_metadata={
                            "data_model:1": DataModelMetadata(
                                datasets_labels=DatasetsLabels(
                                    datasets_labels={
                                        "dataset3": "DATASET3",
                                        "dataset4": "DATASET4",
                                    }
                                ),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset3": "DATASET3",
                                                "dataset4": "DATASET4",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=["tag2"],
                                    properties={
                                        "key2": "value2",
                                        "common-key": "different-value2",
                                    },
                                ),
                            ),
                        }
                    ),
                    "localworker3": DataModelsMetadata(
                        data_models_metadata={
                            "data_model:1": DataModelMetadata(
                                datasets_labels=DatasetsLabels(
                                    datasets_labels={
                                        "dataset5": "DATASET5",
                                        "dataset6": "DATASET6",
                                    }
                                ),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset5": "DATASET5",
                                                "dataset6": "DATASET6",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=["tag3"],
                                    properties={"key3": "value3"},
                                ),
                            ),
                            "data_model:2": DataModelMetadata(
                                datasets_labels=DatasetsLabels(
                                    datasets_labels={
                                        "dataset2": "DATASET2",
                                    }
                                ),
                                cdes=CommonDataElements(
                                    values={
                                        "dataset": CommonDataElement(
                                            code="dataset",
                                            label="Dataset",
                                            sql_type="text",
                                            is_categorical=True,
                                            enumerations={
                                                "dataset2": "DATASET2",
                                            },
                                            min=None,
                                            max=None,
                                        ),
                                    }
                                ),
                                attributes=DataModelAttributes(
                                    tags=["tag2"],
                                    properties={"key2": "value2"},
                                ),
                            ),
                        }
                    ),
                }
            ),
            DataModelRegistry(
                data_models_attributes=DataModelsAttributes(
                    data_models_attributes={
                        "data_model:1": DataModelAttributes(
                            tags=["tag1", "tag2", "tag3"],
                            properties={
                                "key1": ["value1"],
                                "common-key": ["different-value1", "different-value2"],
                                "key2": ["value2"],
                                "key3": ["value3"],
                            },
                        ),
                        "data_model:2": DataModelAttributes(
                            tags=["tag1", "tag2"],
                            properties={
                                "key1": ["value1"],
                                "key2": ["value2"],
                            },
                        ),
                    }
                ),
                data_models_cdes=DataModelsCDES(
                    data_models_cdes={
                        "data_model:1": CommonDataElements(
                            values={
                                "dataset": CommonDataElement(
                                    code="dataset",
                                    label="Dataset",
                                    sql_type="text",
                                    is_categorical=True,
                                    enumerations={
                                        "dataset1": "DATASET1",
                                        "dataset2": "DATASET2",
                                        "dataset3": "DATASET3",
                                        "dataset4": "DATASET4",
                                        "dataset5": "DATASET5",
                                        "dataset6": "DATASET6",
                                    },
                                    min=None,
                                    max=None,
                                )
                            }
                        ),
                        "data_model:2": CommonDataElements(
                            values={
                                "dataset": CommonDataElement(
                                    code="dataset",
                                    label="Dataset",
                                    sql_type="text",
                                    is_categorical=True,
                                    enumerations={
                                        "dataset1": "DATASET1",
                                        "dataset2": "DATASET2",
                                    },
                                    min=None,
                                    max=None,
                                )
                            }
                        ),
                    }
                ),
                datasets_locations=DatasetsLocations(
                    datasets_locations={
                        "data_model:1": {
                            "dataset1": "localworker1",
                            "dataset2": "localworker1",
                            "dataset3": "localworker2",
                            "dataset4": "localworker2",
                            "dataset5": "localworker3",
                            "dataset6": "localworker3",
                        },
                        "data_model:2": {
                            "dataset1": "localworker1",
                            "dataset2": "localworker3",
                        },
                    }
                ),
            ),
            id="properties_with_common_keys",
        ),
    ]


@pytest.mark.parametrize(
    "data_models_metadata_per_worker,expected",
    get_parametrization_cases(),
)
def test_data_model_registry(
    data_models_metadata_per_worker: DataModelsMetadataPerWorker,
    expected: DataModelRegistry,
    worker_landscape_aggregator,
):
    dmr = _crunch_data_model_registry_data(
        data_models_metadata_per_worker, worker_landscape_aggregator._logger
    )
    assert (
        dmr.data_models_cdes.data_models_cdes
        == expected.data_models_cdes.data_models_cdes
    )
    assert (
        dmr.datasets_locations.datasets_locations
        == expected.datasets_locations.datasets_locations
    )
    assert (
        dmr.data_models_attributes.data_models_attributes
        == expected.data_models_attributes.data_models_attributes
    )


def test_data_model_registry_missing_data_model_attributes(worker_landscape_aggregator):
    data_models_metadata_per_worker = DataModelsMetadataPerWorker(
        data_models_metadata_per_worker={
            "localworker1": DataModelsMetadata(
                data_models_metadata={
                    "data_model:1": DataModelMetadata(
                        datasets_labels=DatasetsLabels(
                            datasets_labels={
                                "dataset1": "DATASET1",
                                "dataset2": "DATASET2",
                            }
                        ),
                        cdes=CommonDataElements(
                            values={
                                "dataset": CommonDataElement(
                                    code="dataset",
                                    label="Dataset",
                                    sql_type="text",
                                    is_categorical=True,
                                    enumerations={
                                        "dataset1": "DATASET1",
                                        "dataset2": "DATASET2",
                                    },
                                    min=None,
                                    max=None,
                                ),
                            }
                        ),
                        attributes=None,
                    ),
                    "data_model:2": DataModelMetadata(
                        datasets_labels=DatasetsLabels(
                            datasets_labels={
                                "dataset1": "DATASET1",
                            }
                        ),
                        cdes=CommonDataElements(
                            values={
                                "dataset": CommonDataElement(
                                    code="dataset",
                                    label="Dataset",
                                    sql_type="text",
                                    is_categorical=True,
                                    enumerations={
                                        "dataset1": "DATASET1",
                                    },
                                    min=None,
                                    max=None,
                                ),
                            }
                        ),
                        attributes=DataModelAttributes(
                            tags=["tag1"],
                            properties={"key1": "value1"},
                        ),
                    ),
                }
            ),
            "localworker2": DataModelsMetadata(
                data_models_metadata={
                    "data_model:1": DataModelMetadata(
                        datasets_labels=DatasetsLabels(
                            datasets_labels={
                                "dataset3": "DATASET3",
                                "dataset4": "DATASET4",
                            }
                        ),
                        cdes=CommonDataElements(
                            values={
                                "dataset": CommonDataElement(
                                    code="dataset",
                                    label="Dataset",
                                    sql_type="text",
                                    is_categorical=True,
                                    enumerations={
                                        "dataset3": "DATASET3",
                                        "dataset4": "DATASET4",
                                    },
                                    min=None,
                                    max=None,
                                ),
                            }
                        ),
                        attributes=DataModelAttributes(
                            tags=["tag2"],
                            properties={"key2": "value2"},
                        ),
                    ),
                }
            ),
            "localworker3": DataModelsMetadata(
                data_models_metadata={
                    "data_model:1": DataModelMetadata(
                        datasets_labels=DatasetsLabels(
                            datasets_labels={
                                "dataset5": "DATASET5",
                                "dataset6": "DATASET6",
                            }
                        ),
                        cdes=CommonDataElements(
                            values={
                                "dataset": CommonDataElement(
                                    code="dataset",
                                    label="Dataset",
                                    sql_type="text",
                                    is_categorical=True,
                                    enumerations={
                                        "dataset5": "DATASET5",
                                        "dataset6": "DATASET6",
                                    },
                                    min=None,
                                    max=None,
                                ),
                            }
                        ),
                        attributes=DataModelAttributes(
                            tags=["tag3"],
                            properties={"key3": "value3"},
                        ),
                    ),
                    "data_model:2": DataModelMetadata(
                        datasets_labels=DatasetsLabels(
                            datasets_labels={
                                "dataset2": "DATASET2",
                            }
                        ),
                        cdes=CommonDataElements(
                            values={
                                "dataset": CommonDataElement(
                                    code="dataset",
                                    label="Dataset",
                                    sql_type="text",
                                    is_categorical=True,
                                    enumerations={
                                        "dataset2": "DATASET2",
                                    },
                                    min=None,
                                    max=None,
                                ),
                            }
                        ),
                        attributes=DataModelAttributes(
                            tags=["tag2"],
                            properties={"key2": "value2"},
                        ),
                    ),
                }
            ),
        }
    )
    data_model_registry = DataModelRegistry(
        data_models_attributes=DataModelsAttributes(
            data_models_attributes={
                "data_model:1": DataModelAttributes(
                    tags=["tag2", "tag3"],
                    properties={
                        "key2": ["value2"],
                        "key3": ["value3"],
                    },
                ),
                "data_model:2": DataModelAttributes(
                    tags=["tag1", "tag2"],
                    properties={
                        "key1": ["value1"],
                        "key2": ["value2"],
                    },
                ),
            }
        ),
        data_models_cdes=DataModelsCDES(
            data_models_cdes={
                "data_model:1": CommonDataElements(
                    values={
                        "dataset": CommonDataElement(
                            code="dataset",
                            label="Dataset",
                            sql_type="text",
                            is_categorical=True,
                            enumerations={
                                "dataset1": "DATASET1",
                                "dataset2": "DATASET2",
                                "dataset3": "DATASET3",
                                "dataset4": "DATASET4",
                                "dataset5": "DATASET5",
                                "dataset6": "DATASET6",
                            },
                            min=None,
                            max=None,
                        )
                    }
                ),
                "data_model:2": CommonDataElements(
                    values={
                        "dataset": CommonDataElement(
                            code="dataset",
                            label="Dataset",
                            sql_type="text",
                            is_categorical=True,
                            enumerations={
                                "dataset1": "DATASET1",
                                "dataset2": "DATASET2",
                            },
                            min=None,
                            max=None,
                        )
                    }
                ),
            }
        ),
        datasets_locations=DatasetsLocations(
            datasets_locations={
                "data_model:1": {
                    "dataset3": "localworker2",
                    "dataset4": "localworker2",
                    "dataset5": "localworker3",
                    "dataset6": "localworker3",
                },
                "data_model:2": {
                    "dataset1": "localworker1",
                    "dataset2": "localworker3",
                },
            }
        ),
    )
    dmr = _crunch_data_model_registry_data(
        data_models_metadata_per_worker, worker_landscape_aggregator._logger
    )
    assert (
        dmr.data_models_cdes.data_models_cdes
        == data_model_registry.data_models_cdes.data_models_cdes
    )
    assert (
        dmr.datasets_locations.datasets_locations
        == data_model_registry.datasets_locations.datasets_locations
    )
    assert (
        dmr.data_models_attributes.data_models_attributes
        == data_model_registry.data_models_attributes.data_models_attributes
    )


@pytest.mark.slow
def test_get_workers_info_properly_handles_errors(worker_landscape_aggregator):
    workers_info = worker_landscape_aggregator._get_workers_info(
        [RABBITMQ_LOCALWORKERTMP_ADDR]
    )
    assert not workers_info


@pytest.mark.slow
def test_get_worker_datasets_per_data_model_properly_handles_errors(
    worker_landscape_aggregator,
):
    datasets_per_data_model = (
        worker_landscape_aggregator._get_worker_datasets_per_data_model(
            RABBITMQ_LOCALWORKERTMP_ADDR
        )
    )
    assert not datasets_per_data_model


@pytest.mark.slow
def test_get_worker_cdes_properly_handles_errors(worker_landscape_aggregator):
    cdes = worker_landscape_aggregator._get_worker_cdes(
        RABBITMQ_LOCALWORKERTMP_ADDR, "dementia:0.1"
    )
    assert not cdes

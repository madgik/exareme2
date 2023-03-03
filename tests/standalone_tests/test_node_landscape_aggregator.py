import pytest

from mipengine import AttrDict
from mipengine.controller.node_landscape_aggregator import DataModelMetadata
from mipengine.controller.node_landscape_aggregator import DataModelRegistry
from mipengine.controller.node_landscape_aggregator import DataModelsAttributes
from mipengine.controller.node_landscape_aggregator import DataModelsCDES
from mipengine.controller.node_landscape_aggregator import DataModelsMetadata
from mipengine.controller.node_landscape_aggregator import DataModelsMetadataPerNode
from mipengine.controller.node_landscape_aggregator import DatasetsLabels
from mipengine.controller.node_landscape_aggregator import DatasetsLocations
from mipengine.controller.node_landscape_aggregator import (
    InitializationParams as NodeLandscapeAggregatorInitParams,
)
from mipengine.controller.node_landscape_aggregator import NodeLandscapeAggregator
from mipengine.controller.node_landscape_aggregator import (
    _crunch_data_model_registry_data,
)
from mipengine.node_tasks_DTOs import CommonDataElement
from mipengine.node_tasks_DTOs import CommonDataElements
from mipengine.node_tasks_DTOs import DataModelAttributes
from tests.standalone_tests.conftest import RABBITMQ_LOCALNODETMP_ADDR


@pytest.fixture
def controller_config():
    controller_config = {
        "deployment_type": "LOCAL",
        "node_landscape_aggregator_update_interval": 30,
        "rabbitmq": {
            "celery_tasks_timeout": 5,
            "celery_run_udf_task_timeout": 10,
        },
        "localnodes": {
            "config_file": None,
        },
    }
    return controller_config


@pytest.fixture(scope="function")
def node_landscape_aggregator(
    controller_config,
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
    return node_landscape_aggregator


def get_parametrization_cases():
    return [
        pytest.param(
            DataModelsMetadataPerNode(
                data_models_metadata_per_node={
                    "localnode1": DataModelsMetadata(
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
                    "localnode2": DataModelsMetadata(
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
                    "localnode3": DataModelsMetadata(
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
                            "dataset1": "localnode1",
                            "dataset2": "localnode1",
                            "dataset3": "localnode2",
                            "dataset4": "localnode2",
                            "dataset5": "localnode3",
                            "dataset6": "localnode3",
                        },
                        "data_model:2": {
                            "dataset1": "localnode1",
                            "dataset2": "localnode3",
                        },
                    }
                ),
            ),
            id="common_case",
        ),
        pytest.param(
            DataModelsMetadataPerNode(
                data_models_metadata_per_node={
                    "localnode1": DataModelsMetadata(
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
                    "localnode2": DataModelsMetadata(
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
                    "localnode3": DataModelsMetadata(
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
                            "dataset3": "localnode2",
                            "dataset4": "localnode2",
                            "dataset5": "localnode3",
                            "dataset6": "localnode3",
                        },
                        "data_model:2": {
                            "dataset1": "localnode1",
                            "dataset2": "localnode3",
                        },
                    }
                ),
            ),
            id="none_cdes_on_data_model_1",
        ),
        pytest.param(
            DataModelsMetadataPerNode(
                data_models_metadata_per_node={
                    "localnode1": DataModelsMetadata(
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
                    "localnode2": DataModelsMetadata(
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
                    "localnode3": DataModelsMetadata(
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
                            "dataset1": "localnode1",
                            "dataset2": "localnode3",
                        },
                    }
                ),
            ),
            id="incompatible_cdes_on_data_model1_on_node1_and_node3",
        ),
        pytest.param(
            DataModelsMetadataPerNode(
                data_models_metadata_per_node={
                    "localnode1": DataModelsMetadata(
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
                    "localnode2": DataModelsMetadata(
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
                    "localnode3": DataModelsMetadata(
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
            DataModelsMetadataPerNode(
                data_models_metadata_per_node={
                    "localnode1": DataModelsMetadata(data_models_metadata={}),
                    "localnode2": DataModelsMetadata(data_models_metadata={}),
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
            DataModelsMetadataPerNode(
                data_models_metadata_per_node={
                    "localnode1": DataModelsMetadata(
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
                    "localnode2": DataModelsMetadata(
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
                    "localnode3": DataModelsMetadata(
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
                    "localnode4": DataModelsMetadata(
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
            id="same_data_models_and_datasets_on_all_nodes",
        ),
        pytest.param(
            DataModelsMetadataPerNode(
                data_models_metadata_per_node={
                    "localnode1": DataModelsMetadata(
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
                    "localnode2": DataModelsMetadata(
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
                    "localnode3": DataModelsMetadata(
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
                            "dataset1": "localnode1",
                            "dataset2": "localnode1",
                            "dataset3": "localnode2",
                            "dataset4": "localnode2",
                            "dataset5": "localnode3",
                            "dataset6": "localnode3",
                        },
                        "data_model:2": {},
                    }
                ),
            ),
            id="duplicated_dataset1_on_data_model2",
        ),
        pytest.param(
            DataModelsMetadataPerNode(
                data_models_metadata_per_node={
                    "localnode1": DataModelsMetadata(
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
                    "localnode2": DataModelsMetadata(
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
                    "localnode3": DataModelsMetadata(
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
                            "dataset1": "localnode1",
                            "dataset2": "localnode3",
                        },
                    }
                ),
            ),
            id="incompatible_cdes_on_data_model1",
        ),
        pytest.param(
            DataModelsMetadataPerNode(
                data_models_metadata_per_node={
                    "localnode1": DataModelsMetadata(
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
                    "localnode2": DataModelsMetadata(
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
                    "localnode3": DataModelsMetadata(
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
                            "dataset1": "localnode1",
                            "dataset2": "localnode3",
                        },
                    }
                ),
            ),
            id="incompatible_cdes_on_data_model1",
        ),
        pytest.param(
            DataModelsMetadataPerNode(
                data_models_metadata_per_node={
                    "localnode1": DataModelsMetadata(
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
                    "localnode2": DataModelsMetadata(
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
                    "localnode3": DataModelsMetadata(
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
                            "dataset1": "localnode1",
                            "dataset2": "localnode1",
                            "dataset3": "localnode2",
                            "dataset4": "localnode2",
                            "dataset5": "localnode3",
                            "dataset6": "localnode3",
                        },
                        "data_model:2": {
                            "dataset1": "localnode1",
                            "dataset2": "localnode3",
                        },
                    }
                ),
            ),
            id="duplicated_tags_and_properties",
        ),
        pytest.param(
            DataModelsMetadataPerNode(
                data_models_metadata_per_node={
                    "localnode1": DataModelsMetadata(
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
                    "localnode2": DataModelsMetadata(
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
                    "localnode3": DataModelsMetadata(
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
                            "dataset1": "localnode1",
                            "dataset2": "localnode1",
                            "dataset3": "localnode2",
                            "dataset4": "localnode2",
                            "dataset5": "localnode3",
                            "dataset6": "localnode3",
                        },
                        "data_model:2": {
                            "dataset1": "localnode1",
                            "dataset2": "localnode3",
                        },
                    }
                ),
            ),
            id="properties_with_common_keys",
        ),
    ]


@pytest.mark.parametrize(
    "data_models_metadata_per_node,expected",
    get_parametrization_cases(),
)
def test_data_model_registry(
    data_models_metadata_per_node: DataModelsMetadataPerNode,
    expected: DataModelRegistry,
    node_landscape_aggregator,
):
    dmr = _crunch_data_model_registry_data(
        data_models_metadata_per_node, node_landscape_aggregator._logger
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


@pytest.mark.slow
def test_get_nodes_info_properly_handles_errors(node_landscape_aggregator):
    nodes_info = node_landscape_aggregator._get_nodes_info([RABBITMQ_LOCALNODETMP_ADDR])
    assert not nodes_info


@pytest.mark.slow
def test_get_node_datasets_per_data_model_properly_handles_errors(
    node_landscape_aggregator,
):
    datasets_per_data_model = (
        node_landscape_aggregator._get_node_datasets_per_data_model(
            RABBITMQ_LOCALNODETMP_ADDR
        )
    )
    assert not datasets_per_data_model


@pytest.mark.slow
def test_get_node_cdes_properly_handles_errors(node_landscape_aggregator):
    cdes = node_landscape_aggregator._get_node_cdes(
        RABBITMQ_LOCALNODETMP_ADDR, "dementia:0.1"
    )
    assert not cdes

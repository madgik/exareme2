import pytest

from mipengine.controller.node_landscape_aggregator import DataModelRegistry
from mipengine.controller.node_landscape_aggregator import DataModelsCDES
from mipengine.controller.node_landscape_aggregator import DataModelsMetadata
from mipengine.controller.node_landscape_aggregator import DataModelsMetadataPerNode
from mipengine.controller.node_landscape_aggregator import DatasetsLabels
from mipengine.controller.node_landscape_aggregator import DatasetsLocations
from mipengine.controller.node_landscape_aggregator import (
    _crunch_data_model_registry_data,
)
from mipengine.controller.node_landscape_aggregator import _get_node_cdes
from mipengine.controller.node_landscape_aggregator import (
    _get_node_datasets_per_data_model,
)
from mipengine.controller.node_landscape_aggregator import _get_nodes_info
from mipengine.node_tasks_DTOs import CommonDataElement
from mipengine.node_tasks_DTOs import CommonDataElements
from tests.standalone_tests.conftest import RABBITMQ_LOCALNODETMP_ADDR


def get_parametrization_cases():
    return [
        pytest.param(
            DataModelsMetadataPerNode(
                data_models_metadata_per_node={
                    "localnode1": DataModelsMetadata(
                        data_models_metadata={
                            "data_model:1": (
                                DatasetsLabels(
                                    datasets_labels={
                                        "dataset1": "DATASET1",
                                        "dataset2": "DATASET2",
                                    }
                                ),
                                CommonDataElements(
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
                            ),
                            "data_model:2": (
                                DatasetsLabels(
                                    datasets_labels={
                                        "dataset1": "DATASET1",
                                    }
                                ),
                                CommonDataElements(
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
                            ),
                        }
                    ),
                    "localnode2": DataModelsMetadata(
                        data_models_metadata={
                            "data_model:1": (
                                DatasetsLabels(
                                    datasets_labels={
                                        "dataset3": "DATASET3",
                                        "dataset4": "DATASET4",
                                    }
                                ),
                                CommonDataElements(
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
                            ),
                        }
                    ),
                    "localnode3": DataModelsMetadata(
                        data_models_metadata={
                            "data_model:1": (
                                DatasetsLabels(
                                    datasets_labels={
                                        "dataset5": "DATASET5",
                                        "dataset6": "DATASET6",
                                    }
                                ),
                                CommonDataElements(
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
                            ),
                            "data_model:2": (
                                DatasetsLabels(
                                    datasets_labels={
                                        "dataset2": "DATASET2",
                                    }
                                ),
                                CommonDataElements(
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
                            ),
                        }
                    ),
                }
            ),
            DataModelRegistry(
                data_models=DataModelsCDES(
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
                            "data_model:1": (
                                DatasetsLabels(
                                    datasets_labels={
                                        "dataset1": "DATASET1",
                                        "dataset2": "DATASET2",
                                    }
                                ),
                                None,
                            ),
                            "data_model:2": (
                                DatasetsLabels(
                                    datasets_labels={
                                        "dataset1": "DATASET1",
                                    }
                                ),
                                CommonDataElements(
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
                            ),
                        }
                    ),
                    "localnode2": DataModelsMetadata(
                        data_models_metadata={
                            "data_model:1": (
                                DatasetsLabels(
                                    datasets_labels={
                                        "dataset3": "DATASET3",
                                        "dataset4": "DATASET4",
                                    }
                                ),
                                CommonDataElements(
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
                            ),
                        }
                    ),
                    "localnode3": DataModelsMetadata(
                        data_models_metadata={
                            "data_model:1": (
                                DatasetsLabels(
                                    datasets_labels={
                                        "dataset5": "DATASET5",
                                        "dataset6": "DATASET6",
                                    }
                                ),
                                CommonDataElements(
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
                            ),
                            "data_model:2": (
                                DatasetsLabels(
                                    datasets_labels={
                                        "dataset2": "DATASET2",
                                    }
                                ),
                                CommonDataElements(
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
                            ),
                        }
                    ),
                }
            ),
            DataModelRegistry(
                data_models=DataModelsCDES(
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
                            "data_model:1": (
                                DatasetsLabels(
                                    datasets_labels={
                                        "dataset1": "DATASET1",
                                        "dataset2": "DATASET2",
                                    }
                                ),
                                CommonDataElements(
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
                            ),
                            "data_model:2": (
                                DatasetsLabels(
                                    datasets_labels={
                                        "dataset1": "DATASET1",
                                    }
                                ),
                                CommonDataElements(
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
                            ),
                        }
                    ),
                    "localnode2": DataModelsMetadata(
                        data_models_metadata={
                            "data_model:1": (
                                DatasetsLabels(
                                    datasets_labels={
                                        "dataset3": "DATASET3",
                                        "dataset4": "DATASET4",
                                    }
                                ),
                                CommonDataElements(
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
                            ),
                        }
                    ),
                    "localnode3": DataModelsMetadata(
                        data_models_metadata={
                            "data_model:1": (
                                DatasetsLabels(
                                    datasets_labels={
                                        "dataset5": "DATASET5",
                                        "dataset6": "DATASET6",
                                    }
                                ),
                                CommonDataElements(
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
                            ),
                            "data_model:2": (
                                DatasetsLabels(
                                    datasets_labels={
                                        "dataset2": "DATASET2",
                                    }
                                ),
                                CommonDataElements(
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
                            ),
                        }
                    ),
                }
            ),
            DataModelRegistry(
                data_models=DataModelsCDES(
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
                            "data_model:1": (
                                DatasetsLabels(datasets_labels={}),
                                CommonDataElements(
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
                            ),
                            "data_model:2": (
                                DatasetsLabels(datasets_labels={}),
                                CommonDataElements(
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
                            ),
                        }
                    ),
                    "localnode2": DataModelsMetadata(
                        data_models_metadata={
                            "data_model:1": (
                                DatasetsLabels(datasets_labels={}),
                                CommonDataElements(
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
                            ),
                        }
                    ),
                    "localnode3": DataModelsMetadata(
                        data_models_metadata={
                            "data_model:1": (
                                DatasetsLabels(datasets_labels={}),
                                CommonDataElements(
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
                            ),
                            "data_model:2": (
                                DatasetsLabels(datasets_labels={}),
                                CommonDataElements(
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
                            ),
                        }
                    ),
                }
            ),
            DataModelRegistry(
                data_models=DataModelsCDES(
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
                data_models=DataModelsCDES(data_models_cdes={}),
                datasets_locations=DatasetsLocations(datasets_locations={}),
            ),
            id="no_data_model_or_dataset_case",
        ),
        pytest.param(
            DataModelsMetadataPerNode(
                data_models_metadata_per_node={
                    "localnode1": DataModelsMetadata(
                        data_models_metadata={
                            "data_model:1": (
                                DatasetsLabels(
                                    datasets_labels={
                                        "dataset1": "DATASET1",
                                        "dataset2": "DATASET2",
                                    }
                                ),
                                CommonDataElements(
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
                            ),
                            "data_model:2": (
                                DatasetsLabels(
                                    datasets_labels={
                                        "dataset1": "DATASET1",
                                    }
                                ),
                                CommonDataElements(
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
                            ),
                        }
                    ),
                    "localnode2": DataModelsMetadata(
                        data_models_metadata={
                            "data_model:1": (
                                DatasetsLabels(
                                    datasets_labels={
                                        "dataset1": "DATASET1",
                                        "dataset2": "DATASET2",
                                    }
                                ),
                                CommonDataElements(
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
                            ),
                            "data_model:2": (
                                DatasetsLabels(
                                    datasets_labels={
                                        "dataset1": "DATASET1",
                                    }
                                ),
                                CommonDataElements(
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
                            ),
                        }
                    ),
                    "localnode3": DataModelsMetadata(
                        data_models_metadata={
                            "data_model:1": (
                                DatasetsLabels(
                                    datasets_labels={
                                        "dataset1": "DATASET1",
                                        "dataset2": "DATASET2",
                                    }
                                ),
                                CommonDataElements(
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
                            ),
                            "data_model:2": (
                                DatasetsLabels(
                                    datasets_labels={
                                        "dataset1": "DATASET1",
                                    }
                                ),
                                CommonDataElements(
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
                            ),
                        }
                    ),
                    "localnode4": DataModelsMetadata(
                        data_models_metadata={
                            "data_model:1": (
                                DatasetsLabels(
                                    datasets_labels={
                                        "dataset1": "DATASET1",
                                        "dataset2": "DATASET2",
                                    }
                                ),
                                CommonDataElements(
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
                            ),
                            "data_model:2": (
                                DatasetsLabels(
                                    datasets_labels={
                                        "dataset1": "DATASET1",
                                    }
                                ),
                                CommonDataElements(
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
                            ),
                        }
                    ),
                }
            ),
            DataModelRegistry(
                data_models=DataModelsCDES(
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
                            "data_model:1": (
                                DatasetsLabels(
                                    datasets_labels={
                                        "dataset1": "DATASET1",
                                        "dataset2": "DATASET2",
                                    }
                                ),
                                CommonDataElements(
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
                            ),
                            "data_model:2": (
                                DatasetsLabels(
                                    datasets_labels={
                                        "dataset1": "DATASET1",
                                    }
                                ),
                                CommonDataElements(
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
                            ),
                        }
                    ),
                    "localnode2": DataModelsMetadata(
                        data_models_metadata={
                            "data_model:1": (
                                DatasetsLabels(
                                    datasets_labels={
                                        "dataset3": "DATASET3",
                                        "dataset4": "DATASET4",
                                    }
                                ),
                                CommonDataElements(
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
                            ),
                        }
                    ),
                    "localnode3": DataModelsMetadata(
                        data_models_metadata={
                            "data_model:1": (
                                DatasetsLabels(
                                    datasets_labels={
                                        "dataset5": "DATASET5",
                                        "dataset6": "DATASET6",
                                    }
                                ),
                                CommonDataElements(
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
                            ),
                            "data_model:2": (
                                DatasetsLabels(
                                    datasets_labels={
                                        "dataset1": "DATASET1",
                                    }
                                ),
                                CommonDataElements(
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
                            ),
                        }
                    ),
                }
            ),
            DataModelRegistry(
                data_models=DataModelsCDES(
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
                            "data_model:1": (
                                DatasetsLabels(
                                    datasets_labels={
                                        "dataset1": "DATASET1",
                                        "dataset2": "DATASET2",
                                    }
                                ),
                                CommonDataElements(
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
                            ),
                            "data_model:2": (
                                DatasetsLabels(
                                    datasets_labels={
                                        "dataset1": "DATASET1",
                                    }
                                ),
                                CommonDataElements(
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
                            ),
                        }
                    ),
                    "localnode2": DataModelsMetadata(
                        data_models_metadata={
                            "data_model:1": (
                                DatasetsLabels(
                                    datasets_labels={
                                        "dataset3": "DATASET3",
                                        "dataset4": "DATASET4",
                                    }
                                ),
                                CommonDataElements(
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
                            ),
                        }
                    ),
                    "localnode3": DataModelsMetadata(
                        data_models_metadata={
                            "data_model:1": (
                                DatasetsLabels(
                                    datasets_labels={
                                        "dataset5": "DATASET5",
                                        "dataset6": "DATASET6",
                                    }
                                ),
                                CommonDataElements(
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
                            ),
                            "data_model:2": (
                                DatasetsLabels(
                                    datasets_labels={
                                        "dataset2": "DATASET2",
                                    }
                                ),
                                CommonDataElements(
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
                            ),
                        }
                    ),
                }
            ),
            DataModelRegistry(
                data_models=DataModelsCDES(
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
                            "data_model:1": (
                                DatasetsLabels(
                                    datasets_labels={
                                        "dataset1": "DATASET1",
                                        "dataset2": "DATASET2",
                                    }
                                ),
                                CommonDataElements(
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
                            ),
                            "data_model:2": (
                                DatasetsLabels(
                                    datasets_labels={
                                        "dataset1": "DATASET1",
                                    }
                                ),
                                CommonDataElements(
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
                            ),
                        }
                    ),
                    "localnode2": DataModelsMetadata(
                        data_models_metadata={
                            "data_model:1": (
                                DatasetsLabels(
                                    datasets_labels={
                                        "dataset3": "DATASET3",
                                        "dataset4": "DATASET4",
                                    }
                                ),
                                CommonDataElements(
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
                            ),
                        }
                    ),
                    "localnode3": DataModelsMetadata(
                        data_models_metadata={
                            "data_model:1": (
                                DatasetsLabels(
                                    datasets_labels={
                                        "dataset5": "DATASET5",
                                        "dataset6": "DATASET6",
                                    }
                                ),
                                CommonDataElements(
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
                            ),
                            "data_model:2": (
                                DatasetsLabels(
                                    datasets_labels={
                                        "dataset2": "DATASET2",
                                    }
                                ),
                                CommonDataElements(
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
                            ),
                        }
                    ),
                }
            ),
            DataModelRegistry(
                data_models=DataModelsCDES(
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
    ]


@pytest.mark.parametrize(
    "data_models_metadata_per_node,expected",
    get_parametrization_cases(),
)
def test_data_model_registry(data_models_metadata_per_node, expected):
    assert _crunch_data_model_registry_data(data_models_metadata_per_node) == expected


@pytest.mark.slow
def test_get_nodes_info_properly_handles_errors():
    nodes_info = _get_nodes_info([RABBITMQ_LOCALNODETMP_ADDR])
    assert not nodes_info


@pytest.mark.slow
def test_get_node_datasets_per_data_model_properly_handles_errors():
    datasets_per_data_model = _get_node_datasets_per_data_model(
        RABBITMQ_LOCALNODETMP_ADDR
    )
    assert not datasets_per_data_model


@pytest.mark.slow
def test_get_node_cdes_properly_handles_errors():
    cdes = _get_node_cdes(RABBITMQ_LOCALNODETMP_ADDR, "dementia:0.1")
    assert not cdes

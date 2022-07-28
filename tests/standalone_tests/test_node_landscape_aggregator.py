import pytest

from mipengine.controller.data_model_registry import DataModelCDES
from mipengine.controller.data_model_registry import DataModelRegistry
from mipengine.controller.data_model_registry import DatasetsLocations
from mipengine.controller.node_landscape_aggregator import DataModelsMetadata
from mipengine.controller.node_landscape_aggregator import DataModelsMetadataPerNode
from mipengine.controller.node_landscape_aggregator import DatasetsLabels
from mipengine.controller.node_landscape_aggregator import (
    _data_model_registry_data_cruncing,
)
from mipengine.node_tasks_DTOs import CommonDataElement
from mipengine.node_tasks_DTOs import CommonDataElements


def get_parametrization_cases():
    return [
        pytest.param(
            DataModelsMetadataPerNode(
                values={
                    "localnode1": DataModelsMetadata(
                        values={
                            "data_model:1": (
                                DatasetsLabels(
                                    values={
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
                                    values={
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
                        values={
                            "data_model:1": (
                                DatasetsLabels(
                                    values={
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
                        values={
                            "data_model:1": (
                                DatasetsLabels(
                                    values={
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
                                    values={
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
                data_models=DataModelCDES(
                    values={
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
                    values={
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
                values={
                    "localnode1": DataModelsMetadata(
                        values={
                            "data_model:1": (
                                DatasetsLabels(
                                    values={
                                        "dataset1": "DATASET1",
                                        "dataset2": "DATASET2",
                                    }
                                ),
                                None,
                            ),
                            "data_model:2": (
                                DatasetsLabels(
                                    values={
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
                        values={
                            "data_model:1": (
                                DatasetsLabels(
                                    values={
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
                        values={
                            "data_model:1": (
                                DatasetsLabels(
                                    values={
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
                                    values={
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
                data_models=DataModelCDES(
                    values={
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
                    values={
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
                values={
                    "localnode1": DataModelsMetadata(
                        values={
                            "data_model:1": (
                                DatasetsLabels(
                                    values={
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
                                    values={
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
                        values={
                            "data_model:1": (
                                DatasetsLabels(
                                    values={
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
                        values={
                            "data_model:1": (
                                DatasetsLabels(
                                    values={
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
                                    values={
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
                data_models=DataModelCDES(
                    values={
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
                    values={
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
                values={
                    "localnode1": DataModelsMetadata(
                        values={
                            "data_model:1": (
                                DatasetsLabels(values={}),
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
                                DatasetsLabels(values={}),
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
                        values={
                            "data_model:1": (
                                DatasetsLabels(values={}),
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
                        values={
                            "data_model:1": (
                                DatasetsLabels(values={}),
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
                                DatasetsLabels(values={}),
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
                data_models=DataModelCDES(
                    values={
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
                    values={
                        "data_model:1": {},
                        "data_model:2": {},
                    }
                ),
            ),
            id="no_data_model_or_dataset_case",
        ),
        pytest.param(
            DataModelsMetadataPerNode(
                values={
                    "localnode1": DataModelsMetadata(values={}),
                    "localnode2": DataModelsMetadata(values={}),
                }
            ),
            DataModelRegistry(
                data_models=DataModelCDES(values={}),
                datasets_locations=DatasetsLocations(values={}),
            ),
            id="no_data_model_or_dataset_case",
        ),
        pytest.param(
            DataModelsMetadataPerNode(
                values={
                    "localnode1": DataModelsMetadata(
                        values={
                            "data_model:1": (
                                DatasetsLabels(
                                    values={
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
                                    values={
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
                        values={
                            "data_model:1": (
                                DatasetsLabels(
                                    values={
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
                                    values={
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
                        values={
                            "data_model:1": (
                                DatasetsLabels(
                                    values={
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
                                    values={
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
                        values={
                            "data_model:1": (
                                DatasetsLabels(
                                    values={
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
                                    values={
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
                data_models=DataModelCDES(
                    values={
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
                    values={"data_model:1": {}, "data_model:2": {}}
                ),
            ),
            id="same_data_models_and_datasets_on_all_nodes",
        ),
        pytest.param(
            DataModelsMetadataPerNode(
                values={
                    "localnode1": DataModelsMetadata(
                        values={
                            "data_model:1": (
                                DatasetsLabels(
                                    values={
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
                                    values={
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
                        values={
                            "data_model:1": (
                                DatasetsLabels(
                                    values={
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
                        values={
                            "data_model:1": (
                                DatasetsLabels(
                                    values={
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
                                    values={
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
                data_models=DataModelCDES(
                    values={
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
                    values={
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
                values={
                    "localnode1": DataModelsMetadata(
                        values={
                            "data_model:1": (
                                DatasetsLabels(
                                    values={
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
                                    values={
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
                        values={
                            "data_model:1": (
                                DatasetsLabels(
                                    values={
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
                        values={
                            "data_model:1": (
                                DatasetsLabels(
                                    values={
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
                                    values={
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
                data_models=DataModelCDES(
                    values={
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
                    values={
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
                values={
                    "localnode1": DataModelsMetadata(
                        values={
                            "data_model:1": (
                                DatasetsLabels(
                                    values={
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
                                    values={
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
                        values={
                            "data_model:1": (
                                DatasetsLabels(
                                    values={
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
                        values={
                            "data_model:1": (
                                DatasetsLabels(
                                    values={
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
                                    values={
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
                data_models=DataModelCDES(
                    values={
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
                    values={
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
    assert _data_model_registry_data_cruncing(data_models_metadata_per_node) == expected

import pytest

from mipengine.controller.data_model_registry import DataModelRegistry
from mipengine.controller.node_landscape_aggregator import DatamodelInfo
from mipengine.controller.node_landscape_aggregator import DatasetInfo
from mipengine.controller.node_landscape_aggregator import FederationInfo
from mipengine.controller.node_landscape_aggregator import (
    data_model_registry_data_cruncing,
)
from mipengine.node_tasks_DTOs import CommonDataElement
from mipengine.node_tasks_DTOs import CommonDataElements


def get_parametrization_cases():
    parametrization_list = []

    normal_case = FederationInfo(
        values={
            "localnode1": DatamodelInfo(
                values={
                    "data_model:1": (
                        DatasetInfo(
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
                        DatasetInfo(
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
            "localnode2": DatamodelInfo(
                values={
                    "data_model:1": (
                        DatasetInfo(
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
            "localnode3": DatamodelInfo(
                values={
                    "data_model:1": (
                        DatasetInfo(
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
                        DatasetInfo(
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
    )

    expected = DataModelRegistry(
        data_models={
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
                        enumerations={"dataset1": "DATASET1", "dataset2": "DATASET2"},
                        min=None,
                        max=None,
                    )
                }
            ),
        },
        datasets_locations={
            "data_model:1": {
                "dataset1": "localnode1",
                "dataset2": "localnode1",
                "dataset3": "localnode2",
                "dataset4": "localnode2",
                "dataset5": "localnode3",
                "dataset6": "localnode3",
            },
            "data_model:2": {"dataset1": "localnode1", "dataset2": "localnode3"},
        },
    )
    parametrization_list.append((normal_case, expected))

    no_data_model_or_dataset_case = FederationInfo(
        values={
            "localnode1": DatamodelInfo(
                values={
                    "data_model:1": (
                        DatasetInfo(
                            values={}
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
                        DatasetInfo(
                            values={}
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
            "localnode2": DatamodelInfo(
                values={
                    "data_model:1": (
                        DatasetInfo(
                            values={}
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
            "localnode3": DatamodelInfo(
                values={
                    "data_model:1": (
                        DatasetInfo(
                            values={}
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
                        DatasetInfo(
                            values={}
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
    )

    expected = DataModelRegistry(
        data_models={
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
        },
        datasets_locations={
            "data_model:1": {},
            "data_model:2": {},
        },
    )
    parametrization_list.append((no_data_model_or_dataset_case, expected))

    no_data_model_or_dataset_case = FederationInfo(
        values={
            "localnode1": DatamodelInfo(
                values={}
            ),
            "localnode2": DatamodelInfo(
                values={}
            ),
        }
    )

    expected = DataModelRegistry(
        data_models={
        },
        datasets_locations={
        },
    )
    parametrization_list.append((no_data_model_or_dataset_case, expected))

    same_data_models_and_datasets_on_all_nodes = FederationInfo(
        values={
            "localnode1": DatamodelInfo(
                values={
                    "data_model:1": (
                        DatasetInfo(
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
                        DatasetInfo(
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
            "localnode2": DatamodelInfo(
                values={
                    "data_model:1": (
                        DatasetInfo(
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
                        DatasetInfo(
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
            "localnode3": DatamodelInfo(
                values={
                    "data_model:1": (
                        DatasetInfo(
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
                        DatasetInfo(
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
            "localnode4": DatamodelInfo(
                values={
                    "data_model:1": (
                        DatasetInfo(
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
                        DatasetInfo(
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
    )

    expected = DataModelRegistry(
        data_models={
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
        },
        datasets_locations={"data_model:1": {}, "data_model:2": {}},
    )
    parametrization_list.append((same_data_models_and_datasets_on_all_nodes, expected))

    duplicated_dataset1_on_data_model2 = FederationInfo(
        values={
            "localnode1": DatamodelInfo(
                values={
                    "data_model:1": (
                        DatasetInfo(
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
                        DatasetInfo(
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
            "localnode2": DatamodelInfo(
                values={
                    "data_model:1": (
                        DatasetInfo(
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
            "localnode3": DatamodelInfo(
                values={
                    "data_model:1": (
                        DatasetInfo(
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
                        DatasetInfo(
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
    )

    expected = DataModelRegistry(
        data_models={
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
        },
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
        },
    )
    parametrization_list.append((duplicated_dataset1_on_data_model2, expected))

    incompatible_cdes_on_data_model1 = FederationInfo(
        values={
            "localnode1": DatamodelInfo(
                values={
                    "data_model:1": (
                        DatasetInfo(
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
                        DatasetInfo(
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
            "localnode2": DatamodelInfo(
                values={
                    "data_model:1": (
                        DatasetInfo(
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
            "localnode3": DatamodelInfo(
                values={
                    "data_model:1": (
                        DatasetInfo(
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
                        DatasetInfo(
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
    )

    expected = DataModelRegistry(
        data_models={
            "data_model:2": CommonDataElements(
                values={
                    "dataset": CommonDataElement(
                        code="dataset",
                        label="Dataset",
                        sql_type="text",
                        is_categorical=True,
                        enumerations={"dataset1": "DATASET1", "dataset2": "DATASET2"},
                        min=None,
                        max=None,
                    )
                }
            ),
        },
        datasets_locations={
            "data_model:2": {"dataset1": "localnode1", "dataset2": "localnode3"},
        },
    )
    parametrization_list.append((incompatible_cdes_on_data_model1, expected))

    return parametrization_list


@pytest.mark.parametrize(
    "federated_node_infos,expected",
    get_parametrization_cases(),
)
def test_data_model_registry(federated_node_infos, expected):
    print(f"{data_model_registry_data_cruncing(federated_node_infos)=}")
    print(f"{expected=}")
    assert data_model_registry_data_cruncing(federated_node_infos) == expected

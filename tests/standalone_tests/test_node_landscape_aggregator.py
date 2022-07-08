import pytest

from mipengine.controller.node_landscape_aggregator import _get_compatible_data_models
from mipengine.controller.node_landscape_aggregator import (
    remove_corrupted_data_models_per_node,
)
from mipengine.controller.node_landscape_aggregator import remove_duplicated_datasets
from mipengine.node_tasks_DTOs import CommonDataElement
from mipengine.node_tasks_DTOs import CommonDataElements


def get_parametrization_success_cases():
    parametrization_list = []

    success_case_with_identical_cdes = {
        "dementia:0.1": [
            (
                "localnode1",
                CommonDataElements(
                    values={
                        "dataset": CommonDataElement(
                            code="dataset",
                            label="Dataset",
                            sql_type="text",
                            is_categorical=True,
                            enumerations={
                                "ppmi": "PPMI",
                                "edsd": "EDSD",
                                "desd-synthdata": "DESD-synthdata",
                            },
                        ),
                        "alzheimerbroadcategory": CommonDataElement(
                            code="alzheimerbroadcategory",
                            label="Alzheimer Broad Category",
                            sql_type="text",
                            is_categorical=True,
                            enumerations={
                                "AD": "Alzheimer’s disease",
                                "CN": "Cognitively Normal",
                                "Other": "Other",
                                "MCI": "Mild cognitive impairment",
                            },
                        ),
                    }
                ),
            ),
            (
                "localnode2",
                CommonDataElements(
                    values={
                        "dataset": CommonDataElement(
                            code="dataset",
                            label="Dataset",
                            sql_type="text",
                            is_categorical=True,
                            enumerations={
                                "ppmi": "PPMI",
                                "edsd": "EDSD",
                                "desd-synthdata": "DESD-synthdata",
                            },
                        ),
                        "alzheimerbroadcategory": CommonDataElement(
                            code="alzheimerbroadcategory",
                            label="Alzheimer Broad Category",
                            sql_type="text",
                            is_categorical=True,
                            enumerations={
                                "AD": "Alzheimer’s disease",
                                "CN": "Cognitively Normal",
                                "Other": "Other",
                                "MCI": "Mild cognitive impairment",
                            },
                        ),
                    }
                ),
            ),
        ],
        "tbi:0.1": [
            (
                "localnode1",
                CommonDataElements(
                    values={
                        "dataset": CommonDataElement(
                            code="dataset",
                            label="Dataset",
                            sql_type="text",
                            is_categorical=True,
                            enumerations={"dummy_tbi": "Dummy TBI"},
                        ),
                        "age_value": CommonDataElement(
                            code="age_value",
                            label="Age",
                            sql_type="int",
                            is_categorical=False,
                            min=0.0,
                            max=130.0,
                        ),
                    }
                ),
            ),
            (
                "localnode2",
                CommonDataElements(
                    values={
                        "dataset": CommonDataElement(
                            code="dataset",
                            label="Dataset",
                            sql_type="text",
                            is_categorical=True,
                            enumerations={"dummy_tbi": "Dummy TBI"},
                        ),
                        "age_value": CommonDataElement(
                            code="age_value",
                            label="Age",
                            sql_type="int",
                            is_categorical=False,
                            min=0.0,
                            max=130.0,
                        ),
                    }
                ),
            ),
        ],
    }

    expected_with_identical_cdes = {
        "dementia:0.1": CommonDataElements(
            values={
                "dataset": CommonDataElement(
                    code="dataset",
                    label="Dataset",
                    sql_type="text",
                    is_categorical=True,
                    enumerations={
                        "ppmi": "PPMI",
                        "edsd": "EDSD",
                        "desd-synthdata": "DESD-synthdata",
                    },
                ),
                "alzheimerbroadcategory": CommonDataElement(
                    code="alzheimerbroadcategory",
                    label="Alzheimer Broad Category",
                    sql_type="text",
                    is_categorical=True,
                    enumerations={
                        "AD": "Alzheimer’s disease",
                        "CN": "Cognitively Normal",
                        "Other": "Other",
                        "MCI": "Mild cognitive impairment",
                    },
                ),
            }
        ),
        "tbi:0.1": CommonDataElements(
            values={
                "dataset": CommonDataElement(
                    code="dataset",
                    label="Dataset",
                    sql_type="text",
                    is_categorical=True,
                    enumerations={"dummy_tbi": "Dummy TBI"},
                ),
                "age_value": CommonDataElement(
                    code="age_value",
                    label="Age",
                    sql_type="int",
                    is_categorical=False,
                    min=0.0,
                    max=130.0,
                ),
            }
        ),
    }
    parametrization_list.append(
        (success_case_with_identical_cdes, expected_with_identical_cdes)
    )

    success_case_with_different_dataset_enum = {
        "dementia:0.1": [
            (
                "localnode1",
                CommonDataElements(
                    values={
                        "dataset": CommonDataElement(
                            code="dataset",
                            label="Dataset",
                            sql_type="text",
                            is_categorical=True,
                            enumerations={
                                "ppmi": "PPMI",
                                "edsd": "EDSD",
                                "desd-synthdata": "DESD-synthdata",
                            },
                        ),
                        "alzheimerbroadcategory": CommonDataElement(
                            code="alzheimerbroadcategory",
                            label="Alzheimer Broad Category",
                            sql_type="text",
                            is_categorical=True,
                            enumerations={
                                "AD": "Alzheimer’s disease",
                                "CN": "Cognitively Normal",
                                "Other": "Other",
                                "MCI": "Mild cognitive impairment",
                            },
                        ),
                    }
                ),
            ),
            (
                "localnode2",
                CommonDataElements(
                    values={
                        "dataset": CommonDataElement(
                            code="dataset",
                            label="Dataset",
                            sql_type="text",
                            is_categorical=True,
                            enumerations={
                                "ppmi": "PPMI",
                                "edsd": "EDSD",
                            },
                        ),
                        "alzheimerbroadcategory": CommonDataElement(
                            code="alzheimerbroadcategory",
                            label="Alzheimer Broad Category",
                            sql_type="text",
                            is_categorical=True,
                            enumerations={
                                "AD": "Alzheimer’s disease",
                                "CN": "Cognitively Normal",
                                "Other": "Other",
                                "MCI": "Mild cognitive impairment",
                            },
                        ),
                    }
                ),
            ),
        ],
        "tbi:0.1": [
            (
                "localnode1",
                CommonDataElements(
                    values={
                        "dataset": CommonDataElement(
                            code="dataset",
                            label="Dataset",
                            sql_type="text",
                            is_categorical=True,
                            enumerations={"dummy_tbi": "Dummy TBI"},
                        ),
                        "age_value": CommonDataElement(
                            code="age_value",
                            label="Age",
                            sql_type="int",
                            is_categorical=False,
                            min=0.0,
                            max=130.0,
                        ),
                    }
                ),
            ),
            (
                "localnode2",
                CommonDataElements(
                    values={
                        "dataset": CommonDataElement(
                            code="dataset",
                            label="Dataset",
                            sql_type="text",
                            is_categorical=True,
                            enumerations={
                                "dummy_tbi": "Dummy TBI",
                                "dummy_tbi1": "Dummy TBI1",
                            },
                        ),
                        "age_value": CommonDataElement(
                            code="age_value",
                            label="Age",
                            sql_type="int",
                            is_categorical=False,
                            min=0.0,
                            max=130.0,
                        ),
                    }
                ),
            ),
        ],
    }

    expected_with_different_dataset_enum = {
        "dementia:0.1": CommonDataElements(
            values={
                "dataset": CommonDataElement(
                    code="dataset",
                    label="Dataset",
                    sql_type="text",
                    is_categorical=True,
                    enumerations={
                        "ppmi": "PPMI",
                        "edsd": "EDSD",
                        "desd-synthdata": "DESD-synthdata",
                    },
                ),
                "alzheimerbroadcategory": CommonDataElement(
                    code="alzheimerbroadcategory",
                    label="Alzheimer Broad Category",
                    sql_type="text",
                    is_categorical=True,
                    enumerations={
                        "AD": "Alzheimer’s disease",
                        "CN": "Cognitively Normal",
                        "Other": "Other",
                        "MCI": "Mild cognitive impairment",
                    },
                ),
            }
        ),
        "tbi:0.1": CommonDataElements(
            values={
                "dataset": CommonDataElement(
                    code="dataset",
                    label="Dataset",
                    sql_type="text",
                    is_categorical=True,
                    enumerations={"dummy_tbi": "Dummy TBI"},
                ),
                "age_value": CommonDataElement(
                    code="age_value",
                    label="Age",
                    sql_type="int",
                    is_categorical=False,
                    min=0.0,
                    max=130.0,
                ),
            }
        ),
    }
    parametrization_list.append(
        (success_case_with_different_dataset_enum, expected_with_different_dataset_enum)
    )
    return parametrization_list


@pytest.mark.parametrize(
    "nodes_cdes, expected",
    get_parametrization_success_cases(),
)
def test_get_data_models_success(nodes_cdes, expected):
    assert _get_compatible_data_models(nodes_cdes) == expected


def get_parametrization_fail_cases():
    parametrization_list = []
    expected_result = {
        "tbi:0.1": CommonDataElements(
            values={
                "dataset": CommonDataElement(
                    code="dataset",
                    label="Dataset",
                    sql_type="text",
                    is_categorical=True,
                    enumerations={"dummy_tbi": "Dummy TBI"},
                ),
                "age_value": CommonDataElement(
                    code="age_value",
                    label="Age",
                    sql_type="int",
                    is_categorical=False,
                    min=0.0,
                    max=130.0,
                ),
            }
        )
    }

    # In all 3 fail cases the dementia:0.1 has an incompatibility exception so the data_model should not be contained in the result
    incompatible_cdes_keys = {
        "dementia:0.1": [
            (
                "localnode1",
                CommonDataElements(
                    values={
                        "dataset": CommonDataElement(
                            code="dataset",
                            label="Dataset",
                            sql_type="text",
                            is_categorical=True,
                            enumerations={
                                "ppmi": "PPMI",
                                "edsd": "EDSD",
                                "desd-synthdata": "DESD-synthdata",
                            },
                        ),
                        "alzheimerbroadcategory": CommonDataElement(
                            code="alzheimerbroadcategory",
                            label="Alzheimer Broad Category",
                            sql_type="text",
                            is_categorical=True,
                            enumerations={
                                "AD": "Alzheimer’s disease",
                                "CN": "Cognitively Normal",
                                "Other": "Other",
                                "MCI": "Mild cognitive impairment",
                            },
                        ),
                    }
                ),
            ),
            (
                "localnode2",
                CommonDataElements(
                    values={
                        "dataset": CommonDataElement(
                            code="dataset",
                            label="Dataset",
                            sql_type="text",
                            is_categorical=True,
                            enumerations={
                                "ppmi": "PPMI",
                                "edsd": "EDSD",
                                "desd-synthdata": "DESD-synthdata",
                            },
                        ),
                        "invalid_key": CommonDataElement(
                            code="alzheimerbroadcategory",
                            label="Alzheimer Broad Category",
                            sql_type="text",
                            is_categorical=True,
                            enumerations={
                                "AD": "Alzheimer’s disease",
                                "CN": "Cognitively Normal",
                                "Other": "Other",
                                "MCI": "Mild cognitive impairment",
                            },
                        ),
                    }
                ),
            ),
        ],
        "tbi:0.1": [
            (
                "localnode1",
                CommonDataElements(
                    values={
                        "dataset": CommonDataElement(
                            code="dataset",
                            label="Dataset",
                            sql_type="text",
                            is_categorical=True,
                            enumerations={"dummy_tbi": "Dummy TBI"},
                        ),
                        "age_value": CommonDataElement(
                            code="age_value",
                            label="Age",
                            sql_type="int",
                            is_categorical=False,
                            min=0.0,
                            max=130.0,
                        ),
                    }
                ),
            ),
            (
                "localnode2",
                CommonDataElements(
                    values={
                        "dataset": CommonDataElement(
                            code="dataset",
                            label="Dataset",
                            sql_type="text",
                            is_categorical=True,
                            enumerations={
                                "dummy_tbi": "Dummy TBI",
                                "dummy_tbi1": "Dummy TBI1",
                            },
                        ),
                        "age_value": CommonDataElement(
                            code="age_value",
                            label="Age",
                            sql_type="int",
                            is_categorical=False,
                            min=0.0,
                            max=130.0,
                        ),
                    }
                ),
            ),
        ],
    }

    parametrization_list.append((incompatible_cdes_keys, expected_result))

    incompatible_cdes = {
        "dementia:0.1": [
            (
                "localnode1",
                CommonDataElements(
                    values={
                        "dataset": CommonDataElement(
                            code="dataset",
                            label="Dataset",
                            sql_type="text",
                            is_categorical=True,
                            enumerations={
                                "ppmi": "PPMI",
                                "edsd": "EDSD",
                                "desd-synthdata": "DESD-synthdata",
                            },
                        ),
                        "alzheimerbroadcategory": CommonDataElement(
                            code="alzheimerbroadcategory",
                            label="Alzheimer Broad Category",
                            sql_type="text",
                            is_categorical=True,
                            enumerations={
                                "AD": "Alzheimer’s disease",
                                "CN": "Cognitively Normal",
                                "Other": "Other",
                                "MCI": "Mild cognitive impairment",
                            },
                        ),
                    }
                ),
            ),
            (
                "localnode2",
                CommonDataElements(
                    values={
                        "dataset": CommonDataElement(
                            code="dataset",
                            label="Dataset",
                            sql_type="Incompatible",
                            is_categorical=True,
                            enumerations={
                                "ppmi": "PPMI",
                                "edsd": "EDSD",
                                "desd-synthdata": "DESD-synthdata",
                            },
                        ),
                        "alzheimerbroadcategory": CommonDataElement(
                            code="alzheimerbroadcategory",
                            label="Alzheimer Broad Category",
                            sql_type="text",
                            is_categorical=True,
                            enumerations={
                                "AD": "Alzheimer’s disease",
                                "CN": "Cognitively Normal",
                                "Other": "Other",
                                "MCI": "Mild cognitive impairment",
                            },
                        ),
                    }
                ),
            ),
        ],
        "tbi:0.1": [
            (
                "localnode1",
                CommonDataElements(
                    values={
                        "dataset": CommonDataElement(
                            code="dataset",
                            label="Dataset",
                            sql_type="text",
                            is_categorical=True,
                            enumerations={"dummy_tbi": "Dummy TBI"},
                        ),
                        "age_value": CommonDataElement(
                            code="age_value",
                            label="Age",
                            sql_type="int",
                            is_categorical=False,
                            min=0.0,
                            max=130.0,
                        ),
                    }
                ),
            ),
            (
                "localnode2",
                CommonDataElements(
                    values={
                        "dataset": CommonDataElement(
                            code="dataset",
                            label="Dataset",
                            sql_type="text",
                            is_categorical=True,
                            enumerations={
                                "dummy_tbi": "Dummy TBI",
                                "dummy_tbi1": "Dummy TBI1",
                            },
                        ),
                        "age_value": CommonDataElement(
                            code="age_value",
                            label="Age",
                            sql_type="int",
                            is_categorical=False,
                            min=0.0,
                            max=130.0,
                        ),
                    }
                ),
            ),
        ],
    }

    parametrization_list.append((incompatible_cdes, expected_result))

    incompatible_label_on_dataset_cdes = {
        "dementia:0.1": [
            (
                "localnode1",
                CommonDataElements(
                    values={
                        "dataset": CommonDataElement(
                            code="dataset",
                            label="Dataset",
                            sql_type="text",
                            is_categorical=True,
                            enumerations={
                                "ppmi": "PPMI",
                                "edsd": "EDSD",
                                "desd-synthdata": "DESD-synthdata",
                            },
                        ),
                        "alzheimerbroadcategory": CommonDataElement(
                            code="alzheimerbroadcategory",
                            label="Alzheimer Broad Category",
                            sql_type="text",
                            is_categorical=True,
                            enumerations={
                                "AD": "Alzheimer’s disease",
                                "CN": "Cognitively Normal",
                                "Other": "Other",
                                "MCI": "Mild cognitive impairment",
                            },
                        ),
                    }
                ),
            ),
            (
                "localnode2",
                CommonDataElements(
                    values={
                        "dataset": CommonDataElement(
                            code="dataset",
                            label="Incompatible label",
                            sql_type="text",
                            is_categorical=True,
                            enumerations={
                                "ppmi": "PPMI",
                                "edsd": "EDSD",
                                "desd-synthdata": "DESD-synthdata",
                            },
                        ),
                        "alzheimerbroadcategory": CommonDataElement(
                            code="alzheimerbroadcategory",
                            label="Alzheimer Broad Category",
                            sql_type="text",
                            is_categorical=True,
                            enumerations={
                                "AD": "Alzheimer’s disease",
                                "CN": "Cognitively Normal",
                                "Other": "Other",
                                "MCI": "Mild cognitive impairment",
                            },
                        ),
                    }
                ),
            ),
        ],
        "tbi:0.1": [
            (
                "localnode1",
                CommonDataElements(
                    values={
                        "dataset": CommonDataElement(
                            code="dataset",
                            label="Dataset",
                            sql_type="text",
                            is_categorical=True,
                            enumerations={"dummy_tbi": "Dummy TBI"},
                        ),
                        "age_value": CommonDataElement(
                            code="age_value",
                            label="Age",
                            sql_type="int",
                            is_categorical=False,
                            min=0.0,
                            max=130.0,
                        ),
                    }
                ),
            ),
            (
                "localnode2",
                CommonDataElements(
                    values={
                        "dataset": CommonDataElement(
                            code="dataset",
                            label="Dataset",
                            sql_type="text",
                            is_categorical=True,
                            enumerations={
                                "dummy_tbi": "Dummy TBI",
                                "dummy_tbi1": "Dummy TBI1",
                            },
                        ),
                        "age_value": CommonDataElement(
                            code="age_value",
                            label="Age",
                            sql_type="int",
                            is_categorical=False,
                            min=0.0,
                            max=130.0,
                        ),
                    }
                ),
            ),
        ],
    }

    parametrization_list.append((incompatible_label_on_dataset_cdes, expected_result))

    return parametrization_list


@pytest.mark.parametrize(
    "nodes_cdes, expected_result",
    get_parametrization_fail_cases(),
)
def test_get_data_models_fail(nodes_cdes, expected_result):
    assert _get_compatible_data_models(nodes_cdes) == expected_result


parametrization_cases = [
    (
        {
            "localnode1": {
                "dementia:0.1": {
                    "desd-synthdata0": "DESD-synthdata_0",
                    "edsd0": "EDSD_0",
                },
            },
            "localnode2": {
                "dementia:0.1": {
                    "desd-synthdata1": "DESD-synthdata_1",
                    "desd-synthdata2": "DESD-synthdata_2",
                    "edsd1": "EDSD_1",
                    "edsd2": "EDSD_2",
                },
                "tbi:0.1": {
                    "dummy_tbi0": "Dummy TBI0",
                    "edsd0": "EDSD_0",
                    "desd-synthdata0": "DESD-synthdata_0",
                },
            },
        },
        {
            "localnode1": {
                "dementia:0.1": {
                    "desd-synthdata0": "DESD-synthdata_0",
                    "edsd0": "EDSD_0",
                },
            },
            "localnode2": {
                "dementia:0.1": {
                    "desd-synthdata1": "DESD-synthdata_1",
                    "desd-synthdata2": "DESD-synthdata_2",
                    "edsd1": "EDSD_1",
                    "edsd2": "EDSD_2",
                },
                "tbi:0.1": {
                    "dummy_tbi0": "Dummy TBI0",
                    "edsd0": "EDSD_0",
                    "desd-synthdata0": "DESD-synthdata_0",
                },
            },
        },
    ),
    (
        {
            "localnode1": {
                "dementia:0.1": {
                    "desd-synthdata0": "DESD-synthdata_0",
                    "edsd0": "EDSD_0",
                },
            },
            "localnode2": {
                "dementia:0.1": {
                    "desd-synthdata1": "DESD-synthdata_1",
                    "desd-synthdata2": "DESD-synthdata_2",
                    "edsd1": "EDSD_1",
                    "edsd2": "EDSD_2",
                },
            },
        },
        {
            "localnode1": {
                "dementia:0.1": {
                    "desd-synthdata0": "DESD-synthdata_0",
                    "edsd0": "EDSD_0",
                },
            },
            "localnode2": {
                "dementia:0.1": {
                    "desd-synthdata1": "DESD-synthdata_1",
                    "desd-synthdata2": "DESD-synthdata_2",
                    "edsd1": "EDSD_1",
                    "edsd2": "EDSD_2",
                },
            },
        },
    ),
    (
        {
            "localnode1": {
                "dementia:0.1": {
                    "desd-synthdata0": "DESD-synthdata_0",
                    "desd-synthdata1": "DESD-synthdata_1",
                    "edsd0": "EDSD_0",
                },
            },
            "localnode2": {
                "dementia:0.1": {
                    "desd-synthdata1": "DESD-synthdata_1",
                    "desd-synthdata2": "DESD-synthdata_2",
                    "edsd1": "EDSD_1",
                    "edsd2": "EDSD_2",
                },
            },
        },
        {
            "localnode1": {
                "dementia:0.1": {
                    "desd-synthdata0": "DESD-synthdata_0",
                    "edsd0": "EDSD_0",
                },
            },
            "localnode2": {
                "dementia:0.1": {
                    "desd-synthdata2": "DESD-synthdata_2",
                    "edsd1": "EDSD_1",
                    "edsd2": "EDSD_2",
                },
            },
        },
    ),
    (
        {
            "localnode1": {
                "dementia:0.1": {
                    "desd-synthdata0": "DESD-synthdata_0",
                    "edsd0": "EDSD_0",
                },
            },
            "localnode2": {
                "dementia:0.1": {
                    "desd-synthdata0": "DESD-synthdata_0",
                    "desd-synthdata1": "DESD-synthdata_1",
                    "desd-synthdata2": "DESD-synthdata_2",
                    "edsd1": "EDSD_1",
                    "edsd2": "EDSD_2",
                },
            },
        },
        {
            "localnode1": {
                "dementia:0.1": {
                    "edsd0": "EDSD_0",
                },
            },
            "localnode2": {
                "dementia:0.1": {
                    "desd-synthdata2": "DESD-synthdata_2",
                    "edsd1": "EDSD_1",
                    "edsd2": "EDSD_2",
                },
            },
        },
    ),
    (
        {
            "localnode1": {
                "dementia:0.1": {
                    "desd-synthdata0": "DESD-synthdata_0",
                    "desd-synthdata1": "DESD-synthdata_1",
                    "desd-synthdata2": "DESD-synthdata_2",
                    "edsd0": "EDSD_0",
                    "edsd1": "EDSD_1",
                    "edsd2": "EDSD_2",
                },
            },
            "localnode2": {
                "dementia:0.1": {
                    "desd-synthdata0": "DESD-synthdata_0",
                    "desd-synthdata1": "DESD-synthdata_1",
                    "desd-synthdata2": "DESD-synthdata_2",
                    "edsd0": "EDSD_0",
                    "edsd1": "EDSD_1",
                    "edsd2": "EDSD_2",
                },
            },
        },
        {
            "localnode1": {
                "dementia:0.1": {},
            },
            "localnode2": {
                "dementia:0.1": {},
            },
        },
    ),
]


@pytest.mark.parametrize(
    "datasets_per_node, expected_result",
    parametrization_cases,
)
def test_remove_duplicated_datasets(datasets_per_node, expected_result):
    datasets_per_node_without_duplicates = remove_duplicated_datasets(datasets_per_node)
    assert all(
        [
            datasets_per_node_without_duplicates[node_id][data_model][dataset_name]
            == expected_result[node_id][data_model][dataset_name]
            for node_id, datasets_per_data_model in expected_result.items()
            for data_model, datasets in datasets_per_data_model.items()
            for dataset_name, dataset_label in datasets.items()
        ]
    )


def get_cases_remove_corrupted_data_models_per_node():
    parametrization_list = []
    case1 = {
        "dementia:0.1": [
            (
                "localnode1",
                CommonDataElements(
                    values={
                        "dataset": CommonDataElement(
                            code="dataset",
                            label="Dataset",
                            sql_type="text",
                            is_categorical=True,
                            enumerations={
                                "ppmi": "PPMI",
                                "edsd": "EDSD",
                                "desd-synthdata": "DESD-synthdata",
                            },
                        ),
                    }
                ),
            ),
            (
                "localnode2",
                CommonDataElements(
                    values={
                        "dataset": CommonDataElement(
                            code="dataset",
                            label="Dataset",
                            sql_type="text",
                            is_categorical=True,
                            enumerations={
                                "ppmi": "PPMI",
                                "edsd": "EDSD",
                                "desd-synthdata": "DESD-synthdata",
                            },
                        ),
                    }
                ),
            ),
        ],
        "tbi:0.1": [
            (
                "localnode1",
                CommonDataElements(
                    values={
                        "dataset": CommonDataElement(
                            code="dataset",
                            label="Dataset",
                            sql_type="text",
                            is_categorical=True,
                            enumerations={"dummy_tbi": "Dummy TBI"},
                        ),
                    }
                ),
            ),
            (
                "localnode2",
                CommonDataElements(
                    values={
                        "dataset": CommonDataElement(
                            code="dataset",
                            label="Dataset",
                            sql_type="text",
                            is_categorical=True,
                            enumerations={"dummy_tbi": "Dummy TBI"},
                        ),
                    }
                ),
            ),
        ],
    }

    expected = {
        "dementia:0.1": [
            (
                "localnode1",
                CommonDataElements(
                    values={
                        "dataset": CommonDataElement(
                            code="dataset",
                            label="Dataset",
                            sql_type="text",
                            is_categorical=True,
                            enumerations={
                                "ppmi": "PPMI",
                                "edsd": "EDSD",
                                "desd-synthdata": "DESD-synthdata",
                            },
                        ),
                    }
                ),
            ),
            (
                "localnode2",
                CommonDataElements(
                    values={
                        "dataset": CommonDataElement(
                            code="dataset",
                            label="Dataset",
                            sql_type="text",
                            is_categorical=True,
                            enumerations={
                                "ppmi": "PPMI",
                                "edsd": "EDSD",
                                "desd-synthdata": "DESD-synthdata",
                            },
                        ),
                    }
                ),
            ),
        ],
        "tbi:0.1": [
            (
                "localnode1",
                CommonDataElements(
                    values={
                        "dataset": CommonDataElement(
                            code="dataset",
                            label="Dataset",
                            sql_type="text",
                            is_categorical=True,
                            enumerations={"dummy_tbi": "Dummy TBI"},
                        ),
                    }
                ),
            ),
            (
                "localnode2",
                CommonDataElements(
                    values={
                        "dataset": CommonDataElement(
                            code="dataset",
                            label="Dataset",
                            sql_type="text",
                            is_categorical=True,
                            enumerations={"dummy_tbi": "Dummy TBI"},
                        ),
                    }
                ),
            ),
        ],
    }
    parametrization_list.append((case1, expected))
    return parametrization_list


@pytest.mark.parametrize(
    "node_cdes_with_corrupted_data_models,expected_result",
    get_cases_remove_corrupted_data_models_per_node(),
)
def test_remove_corrupted_data_models_per_node(
    node_cdes_with_corrupted_data_models, expected_result
):

    data_model_cdes_per_node_without_corrupted = remove_corrupted_data_models_per_node(
        node_cdes_with_corrupted_data_models
    )

    assert data_model_cdes_per_node_without_corrupted == expected_result

import random
from unittest.mock import patch

import pytest

from mipengine.controller.node_landscape_aggregator import _get_cdes_across_nodes
from mipengine.controller.node_landscape_aggregator import _get_compatible_data_models
from mipengine.controller.node_landscape_aggregator import remove_duplicated_datasets
from mipengine.node_tasks_DTOs import CommonDataElement
from mipengine.node_tasks_DTOs import CommonDataElements
from tests.standalone_tests.test_node_registry import get_mocked_node_info


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
            },
            "localnode3": {
                "dementia:0.1": None,
            },
        },
        {
            "dementia:0.1": [
                (
                    "localnode1",
                    {
                        "dataset": CommonDataElement(
                            code="dataset",
                            label="Dataset",
                            sql_type="text",
                            is_categorical=True,
                            enumerations={
                                "ppmi0": "PPMI_0",
                                "ppmi1": "PPMI_1",
                                "ppmi2": "PPMI_2",
                                "ppmi3": "PPMI_3",
                                "ppmi4": "PPMI_4",
                                "ppmi5": "PPMI_5",
                                "ppmi6": "PPMI_6",
                                "ppmi7": "PPMI_7",
                                "ppmi8": "PPMI_8",
                                "ppmi9": "PPMI_9",
                                "edsd0": "EDSD_0",
                                "edsd1": "EDSD_1",
                                "edsd2": "EDSD_2",
                                "edsd3": "EDSD_3",
                                "edsd4": "EDSD_4",
                                "edsd5": "EDSD_5",
                                "edsd6": "EDSD_6",
                                "edsd7": "EDSD_7",
                                "edsd8": "EDSD_8",
                                "edsd9": "EDSD_9",
                                "desd-synthdata0": "DESD-synthdata_0",
                                "desd-synthdata1": "DESD-synthdata_1",
                                "desd-synthdata2": "DESD-synthdata_2",
                                "desd-synthdata3": "DESD-synthdata_3",
                                "desd-synthdata4": "DESD-synthdata_4",
                                "desd-synthdata5": "DESD-synthdata_5",
                                "desd-synthdata6": "DESD-synthdata_6",
                                "desd-synthdata7": "DESD-synthdata_7",
                                "desd-synthdata8": "DESD-synthdata_8",
                                "desd-synthdata9": "DESD-synthdata_9",
                            },
                            min=None,
                            max=None,
                        ),
                        "subjectvisitid": CommonDataElement(
                            code="subjectvisitid",
                            label="Visit ID",
                            sql_type="text",
                            is_categorical=False,
                            enumerations=None,
                            min=None,
                            max=None,
                        ),
                        "subjectvisitdate": CommonDataElement(
                            code="subjectvisitdate",
                            label="Visit Date",
                            sql_type="text",
                            is_categorical=False,
                            enumerations=None,
                            min=None,
                            max=None,
                        ),
                    },
                ),
                (
                    "localnode2",
                    {
                        "dataset": CommonDataElement(
                            code="dataset",
                            label="Dataset",
                            sql_type="text",
                            is_categorical=True,
                            enumerations={
                                "ppmi0": "PPMI_0",
                                "ppmi1": "PPMI_1",
                                "ppmi2": "PPMI_2",
                                "ppmi3": "PPMI_3",
                                "ppmi4": "PPMI_4",
                                "ppmi5": "PPMI_5",
                                "ppmi6": "PPMI_6",
                                "ppmi7": "PPMI_7",
                                "ppmi8": "PPMI_8",
                                "ppmi9": "PPMI_9",
                                "edsd0": "EDSD_0",
                                "edsd1": "EDSD_1",
                                "edsd2": "EDSD_2",
                                "edsd3": "EDSD_3",
                                "edsd4": "EDSD_4",
                                "edsd5": "EDSD_5",
                                "edsd6": "EDSD_6",
                                "edsd7": "EDSD_7",
                                "edsd8": "EDSD_8",
                                "edsd9": "EDSD_9",
                                "desd-synthdata0": "DESD-synthdata_0",
                                "desd-synthdata1": "DESD-synthdata_1",
                                "desd-synthdata2": "DESD-synthdata_2",
                                "desd-synthdata3": "DESD-synthdata_3",
                                "desd-synthdata4": "DESD-synthdata_4",
                                "desd-synthdata5": "DESD-synthdata_5",
                                "desd-synthdata6": "DESD-synthdata_6",
                                "desd-synthdata7": "DESD-synthdata_7",
                                "desd-synthdata8": "DESD-synthdata_8",
                                "desd-synthdata9": "DESD-synthdata_9",
                            },
                            min=None,
                            max=None,
                        ),
                        "subjectvisitid": CommonDataElement(
                            code="subjectvisitid",
                            label="Visit ID",
                            sql_type="text",
                            is_categorical=False,
                            enumerations=None,
                            min=None,
                            max=None,
                        ),
                        "subjectvisitdate": CommonDataElement(
                            code="subjectvisitdate",
                            label="Visit Date",
                            sql_type="text",
                            is_categorical=False,
                            enumerations=None,
                            min=None,
                            max=None,
                        ),
                    },
                ),
            ]
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
            "dementia:0.1": [
                (
                    "localnode1",
                    {
                        "dataset": CommonDataElement(
                            code="dataset",
                            label="Dataset",
                            sql_type="text",
                            is_categorical=True,
                            enumerations={
                                "ppmi0": "PPMI_0",
                                "ppmi1": "PPMI_1",
                                "ppmi2": "PPMI_2",
                                "ppmi3": "PPMI_3",
                                "ppmi4": "PPMI_4",
                                "ppmi5": "PPMI_5",
                                "ppmi6": "PPMI_6",
                                "ppmi7": "PPMI_7",
                                "ppmi8": "PPMI_8",
                                "ppmi9": "PPMI_9",
                                "edsd0": "EDSD_0",
                                "edsd1": "EDSD_1",
                                "edsd2": "EDSD_2",
                                "edsd3": "EDSD_3",
                                "edsd4": "EDSD_4",
                                "edsd5": "EDSD_5",
                                "edsd6": "EDSD_6",
                                "edsd7": "EDSD_7",
                                "edsd8": "EDSD_8",
                                "edsd9": "EDSD_9",
                                "desd-synthdata0": "DESD-synthdata_0",
                                "desd-synthdata1": "DESD-synthdata_1",
                                "desd-synthdata2": "DESD-synthdata_2",
                                "desd-synthdata3": "DESD-synthdata_3",
                                "desd-synthdata4": "DESD-synthdata_4",
                                "desd-synthdata5": "DESD-synthdata_5",
                                "desd-synthdata6": "DESD-synthdata_6",
                                "desd-synthdata7": "DESD-synthdata_7",
                                "desd-synthdata8": "DESD-synthdata_8",
                                "desd-synthdata9": "DESD-synthdata_9",
                            },
                            min=None,
                            max=None,
                        ),
                        "subjectvisitid": CommonDataElement(
                            code="subjectvisitid",
                            label="Visit ID",
                            sql_type="text",
                            is_categorical=False,
                            enumerations=None,
                            min=None,
                            max=None,
                        ),
                        "subjectvisitdate": CommonDataElement(
                            code="subjectvisitdate",
                            label="Visit Date",
                            sql_type="text",
                            is_categorical=False,
                            enumerations=None,
                            min=None,
                            max=None,
                        ),
                    },
                ),
                (
                    "localnode2",
                    {
                        "dataset": CommonDataElement(
                            code="dataset",
                            label="Dataset",
                            sql_type="text",
                            is_categorical=True,
                            enumerations={
                                "ppmi0": "PPMI_0",
                                "ppmi1": "PPMI_1",
                                "ppmi2": "PPMI_2",
                                "ppmi3": "PPMI_3",
                                "ppmi4": "PPMI_4",
                                "ppmi5": "PPMI_5",
                                "ppmi6": "PPMI_6",
                                "ppmi7": "PPMI_7",
                                "ppmi8": "PPMI_8",
                                "ppmi9": "PPMI_9",
                                "edsd0": "EDSD_0",
                                "edsd1": "EDSD_1",
                                "edsd2": "EDSD_2",
                                "edsd3": "EDSD_3",
                                "edsd4": "EDSD_4",
                                "edsd5": "EDSD_5",
                                "edsd6": "EDSD_6",
                                "edsd7": "EDSD_7",
                                "edsd8": "EDSD_8",
                                "edsd9": "EDSD_9",
                                "desd-synthdata0": "DESD-synthdata_0",
                                "desd-synthdata1": "DESD-synthdata_1",
                                "desd-synthdata2": "DESD-synthdata_2",
                                "desd-synthdata3": "DESD-synthdata_3",
                                "desd-synthdata4": "DESD-synthdata_4",
                                "desd-synthdata5": "DESD-synthdata_5",
                                "desd-synthdata6": "DESD-synthdata_6",
                                "desd-synthdata7": "DESD-synthdata_7",
                                "desd-synthdata8": "DESD-synthdata_8",
                                "desd-synthdata9": "DESD-synthdata_9",
                            },
                            min=None,
                            max=None,
                        ),
                        "subjectvisitid": CommonDataElement(
                            code="subjectvisitid",
                            label="Visit ID",
                            sql_type="text",
                            is_categorical=False,
                            enumerations=None,
                            min=None,
                            max=None,
                        ),
                        "subjectvisitdate": CommonDataElement(
                            code="subjectvisitdate",
                            label="Visit Date",
                            sql_type="text",
                            is_categorical=False,
                            enumerations=None,
                            min=None,
                            max=None,
                        ),
                    },
                ),
            ]
        },
    ),
]


def _get_node_cdes(node_socket_addr, data_model):
    return {
        "1_dementia:0.1": {
            "dataset": CommonDataElement(
                code="dataset",
                label="Dataset",
                sql_type="text",
                is_categorical=True,
                enumerations={
                    "ppmi0": "PPMI_0",
                    "ppmi1": "PPMI_1",
                    "ppmi2": "PPMI_2",
                    "ppmi3": "PPMI_3",
                    "ppmi4": "PPMI_4",
                    "ppmi5": "PPMI_5",
                    "ppmi6": "PPMI_6",
                    "ppmi7": "PPMI_7",
                    "ppmi8": "PPMI_8",
                    "ppmi9": "PPMI_9",
                    "edsd0": "EDSD_0",
                    "edsd1": "EDSD_1",
                    "edsd2": "EDSD_2",
                    "edsd3": "EDSD_3",
                    "edsd4": "EDSD_4",
                    "edsd5": "EDSD_5",
                    "edsd6": "EDSD_6",
                    "edsd7": "EDSD_7",
                    "edsd8": "EDSD_8",
                    "edsd9": "EDSD_9",
                    "desd-synthdata0": "DESD-synthdata_0",
                    "desd-synthdata1": "DESD-synthdata_1",
                    "desd-synthdata2": "DESD-synthdata_2",
                    "desd-synthdata3": "DESD-synthdata_3",
                    "desd-synthdata4": "DESD-synthdata_4",
                    "desd-synthdata5": "DESD-synthdata_5",
                    "desd-synthdata6": "DESD-synthdata_6",
                    "desd-synthdata7": "DESD-synthdata_7",
                    "desd-synthdata8": "DESD-synthdata_8",
                    "desd-synthdata9": "DESD-synthdata_9",
                },
                min=None,
                max=None,
            ),
            "subjectvisitid": CommonDataElement(
                code="subjectvisitid",
                label="Visit ID",
                sql_type="text",
                is_categorical=False,
                enumerations=None,
                min=None,
                max=None,
            ),
            "subjectvisitdate": CommonDataElement(
                code="subjectvisitdate",
                label="Visit Date",
                sql_type="text",
                is_categorical=False,
                enumerations=None,
                min=None,
                max=None,
            ),
        },
        "2_dementia:0.1": {
            "dataset": CommonDataElement(
                code="dataset",
                label="Dataset",
                sql_type="text",
                is_categorical=True,
                enumerations={
                    "ppmi0": "PPMI_0",
                    "ppmi1": "PPMI_1",
                    "ppmi2": "PPMI_2",
                    "ppmi3": "PPMI_3",
                    "ppmi4": "PPMI_4",
                    "ppmi5": "PPMI_5",
                    "ppmi6": "PPMI_6",
                    "ppmi7": "PPMI_7",
                    "ppmi8": "PPMI_8",
                    "ppmi9": "PPMI_9",
                    "edsd0": "EDSD_0",
                    "edsd1": "EDSD_1",
                    "edsd2": "EDSD_2",
                    "edsd3": "EDSD_3",
                    "edsd4": "EDSD_4",
                    "edsd5": "EDSD_5",
                    "edsd6": "EDSD_6",
                    "edsd7": "EDSD_7",
                    "edsd8": "EDSD_8",
                    "edsd9": "EDSD_9",
                    "desd-synthdata0": "DESD-synthdata_0",
                    "desd-synthdata1": "DESD-synthdata_1",
                    "desd-synthdata2": "DESD-synthdata_2",
                    "desd-synthdata3": "DESD-synthdata_3",
                    "desd-synthdata4": "DESD-synthdata_4",
                    "desd-synthdata5": "DESD-synthdata_5",
                    "desd-synthdata6": "DESD-synthdata_6",
                    "desd-synthdata7": "DESD-synthdata_7",
                    "desd-synthdata8": "DESD-synthdata_8",
                    "desd-synthdata9": "DESD-synthdata_9",
                },
                min=None,
                max=None,
            ),
            "subjectvisitid": CommonDataElement(
                code="subjectvisitid",
                label="Visit ID",
                sql_type="text",
                is_categorical=False,
                enumerations=None,
                min=None,
                max=None,
            ),
            "subjectvisitdate": CommonDataElement(
                code="subjectvisitdate",
                label="Visit Date",
                sql_type="text",
                is_categorical=False,
                enumerations=None,
                min=None,
                max=None,
            ),
        },
        "3_dementia:0.1": None,
    }[f"{node_socket_addr[-1]}_{data_model}"]


@pytest.fixture(scope="module", autouse=True)
def node_cdes():
    with patch(
        "mipengine.controller.node_landscape_aggregator._get_node_cdes",
        side_effect=_get_node_cdes,
    ):
        yield


@pytest.mark.parametrize(
    "datasets_per_node,expected_result",
    parametrization_cases,
)
def test_get_cdes_across_nodes(datasets_per_node, expected_result, node_cdes):
    if "localnode3" in datasets_per_node:
        assert datasets_per_node["localnode3"] == {"dementia:0.1": None}

    cdes_across_nodes = _get_cdes_across_nodes(
        nodes=get_mocked_node_info(), datasets_per_node=datasets_per_node
    )

    if "localnode3" in datasets_per_node:
        assert datasets_per_node["localnode3"] == {}
    assert cdes_across_nodes == expected_result

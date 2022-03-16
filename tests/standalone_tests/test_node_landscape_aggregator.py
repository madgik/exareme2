import pytest

from mipengine.controller.common_data_elements import CommonDataElement
from mipengine.controller.common_data_elements import CommonDataElements
from mipengine.controller.node_landscape_aggregator import _get_common_data_models


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
def test_get_common_data_models_success(nodes_cdes, expected):
    assert _get_common_data_models(nodes_cdes) == expected


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
def test_get_common_data_models_fail(nodes_cdes, expected_result):
    assert _get_common_data_models(nodes_cdes) == expected_result

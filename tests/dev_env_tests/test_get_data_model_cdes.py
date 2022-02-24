import uuid

import pytest

from mipengine.node_tasks_DTOs import CommonDataElement
from tests.dev_env_tests.nodes_communication import get_celery_app
from tests.dev_env_tests.nodes_communication import get_celery_task_signature


@pytest.fixture(scope="session")
def localnode_id():
    return "localnode1"


def get_test_cases_get_data_model_cdes():
    test_cases_get_data_model_cdes = [
        (
            "dementia:0.1",
            186,
            {
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
                "minimentalstate": CommonDataElement(
                    code="minimentalstate",
                    label="MMSE Total scores",
                    sql_type="int",
                    is_categorical=False,
                    min=0,
                    max=30,
                ),
            },
        ),
        (
            "tbi:0.1",
            21,
            {
                "dataset": CommonDataElement(
                    code="dataset",
                    label="Dataset",
                    sql_type="text",
                    is_categorical=True,
                    enumerations={
                        "dummy_tbi": "Dummy TBI",
                    },
                ),
                "gender_type": CommonDataElement(
                    code="gender_type",
                    label="Gender",
                    sql_type="text",
                    is_categorical=True,
                    enumerations={
                        "M": "Male",
                        "F": "Female",
                    },
                ),
            },
        ),
    ]
    return test_cases_get_data_model_cdes


@pytest.mark.parametrize(
    "data_model, expected_data_model_cdes_length, expected_data_model_cdes",
    get_test_cases_get_data_model_cdes(),
)
def test_get_data_model_cdes(
    data_model, expected_data_model_cdes_length, expected_data_model_cdes
):
    node_info_signature = get_celery_task_signature(
        get_celery_app("localnode1"), "get_data_model_cdes"
    )
    request_id = "test_node_info_" + uuid.uuid4().hex + "_request"
    data_model_cdes = node_info_signature.delay(
        request_id=request_id, data_model=data_model
    ).get()

    assert len(data_model_cdes) == expected_data_model_cdes_length
    for exp_cde_code, exp_cde_metadata in expected_data_model_cdes.items():
        assert exp_cde_code in data_model_cdes.keys()

        # The data_model_cde is first parsed into a CommonDataElement and then dumped into a json
        # in order for additional properties to be removed and for the order to be the same.
        assert (
            CommonDataElement.parse_raw(data_model_cdes[exp_cde_code]).json()
            == exp_cde_metadata.json()
        )

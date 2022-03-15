import pytest

from mipengine.controller.common_data_elements import CommonDataElement
from mipengine.controller.data_model_registry import DataModelRegistry


# TODO the testing should be better once the datasets are properly distributed and the are no duplicates.
def get_datasets_location():
    return {
        "tbi:0.1": {"dummy_tbi": ["localnode1", "localnode2"]},
        "dementia:0.1": {
            "ppmi": ["localnode1", "localnode2"],
            "edsd": ["localnode1", "localnode2"],
            "desd-synthdata": ["localnode1", "localnode2"],
        },
    }


def get_data_model_cdes():
    return {
        "dementia:0.1": {
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
        "tbi:0.1": {
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
    }


@pytest.fixture
def mocked_data_model_registry():
    data_model_registry = DataModelRegistry()
    data_model_registry.set_common_data_models(get_data_model_cdes())
    data_model_registry.set_datasets_location(get_datasets_location())
    return data_model_registry


def test_get_all_available_data_models(mocked_data_model_registry):
    assert "dementia:0.1" in mocked_data_model_registry.common_data_models
    assert "tbi:0.1" in mocked_data_model_registry.common_data_models


def test_get_all_available_datasets_per_data_model(mocked_data_model_registry):
    assert mocked_data_model_registry.get_all_available_datasets_per_data_model()


def test_data_model_exists(mocked_data_model_registry):
    assert mocked_data_model_registry.data_model_exists("tbi:0.1")
    assert not mocked_data_model_registry.data_model_exists("non-existing")


def test_dataset_exists(mocked_data_model_registry):
    assert mocked_data_model_registry.dataset_exists("tbi:0.1", "dummy_tbi")
    assert not mocked_data_model_registry.dataset_exists("tbi:0.1", "non-existing")


def test_get_nodes_with_any_of_datasets(mocked_data_model_registry):
    assert set(
        mocked_data_model_registry.get_nodes_with_any_of_datasets(
            "tbi:0.1", ["dummy_tbi"]
        )
    ) == {"localnode1", "localnode2"}

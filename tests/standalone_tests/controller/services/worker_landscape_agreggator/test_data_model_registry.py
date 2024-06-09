import pytest

# TODO the testing should be better once the datasets are properly distributed and the are no duplicates.
from exareme2.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    DataModelRegistry,
)
from exareme2.worker_communication import CommonDataElement
from exareme2.worker_communication import DataModelMetadata


@pytest.fixture
def mocked_datasets_locations():
    yield {
        "tbi:0.1": {
            "dummy_tbi0": "localworker1",
            "dummy_tbi1": "localworker2",
            "dummy_tbi3": "localworker2",
        },
        "dementia:0.1": {
            "ppmi0": "localworker1",
            "ppmi1": "localworker2",
            "ppmi3": "localworker2",
            "edsd0": "localworker1",
            "edsd1": "localworker2",
            "edsd3": "localworker2",
            "desd-synthdata0": "localworker1",
            "desd-synthdata1": "localworker2",
            "desd-synthdata3": "localworker2",
        },
    }


@pytest.fixture
def mocked_data_models_metadata():
    yield {
        "dementia:0.1": DataModelMetadata(
            code="dementia",
            version="0.1",
            label="dementia",
            variables=[
                CommonDataElement(
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
                CommonDataElement(
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
                CommonDataElement(
                    code="minimentalstate",
                    label="MMSE Total scores",
                    sql_type="int",
                    is_categorical=False,
                    min=0,
                    max=30,
                ),
            ],
            groups=[],
            longitudinal=False,
        ),
        "tbi:0.1": DataModelMetadata(
            code="tbi",
            version="0.1",
            label="tbi",
            variables=[
                CommonDataElement(
                    code="dataset",
                    label="Dataset",
                    sql_type="text",
                    is_categorical=True,
                    enumerations={
                        "dummy_tbi": "Dummy TBI",
                    },
                ),
                CommonDataElement(
                    code="gender_type",
                    label="Gender",
                    sql_type="text",
                    is_categorical=True,
                    enumerations={
                        "M": "Male",
                        "F": "Female",
                    },
                ),
            ],
            groups=[],
            longitudinal=True,
        ),
    }


@pytest.fixture
def mocked_data_model_registry(mocked_datasets_locations, mocked_data_models_metadata):
    data_model_registry = DataModelRegistry(
        datasets_locations=mocked_datasets_locations,
        data_models_metadata=mocked_data_models_metadata,
    )
    return data_model_registry


def test_get_all_available_datasets_per_data_model(mocked_data_model_registry):
    assert mocked_data_model_registry.get_all_available_datasets_per_data_model()


def test_data_model_exists(mocked_data_model_registry):
    assert mocked_data_model_registry.data_model_exists("tbi:0.1")
    assert not mocked_data_model_registry.data_model_exists("non-existing")


def test_dataset_exists(mocked_data_model_registry):
    assert mocked_data_model_registry.dataset_exists("tbi:0.1", "dummy_tbi0")
    assert not mocked_data_model_registry.dataset_exists("tbi:0.1", "non-existing")


def test_get_workers_with_any_of_datasets(mocked_data_model_registry):
    assert set(
        mocked_data_model_registry.get_worker_ids_with_any_of_datasets(
            "tbi:0.1", ["dummy_tbi0", "dummy_tbi1"]
        )
    ) == {"localworker1", "localworker2"}


def test_get_worker_specific_datasets(mocked_data_model_registry):
    assert set(
        mocked_data_model_registry.get_worker_specific_datasets(
            "localworker1", "dementia:0.1", ["edsd0", "ppmi0"]
        )
    ) == {"edsd0", "ppmi0"}


def test_is_longitudinal(mocked_data_model_registry):
    assert mocked_data_model_registry.is_longitudinal("tbi:0.1")
    assert not mocked_data_model_registry.is_longitudinal("dementia:0.1")


def test_get_data_models_metadata(mocked_data_model_registry):
    data_models_metadata = mocked_data_model_registry.get_data_models_metadata()
    assert "tbi:0.1" in data_models_metadata
    assert "dementia:0.1" in data_models_metadata
    assert (
        DataModelMetadata(
            code="tbi",
            version="0.1",
            label="tbi",
            variables=[
                CommonDataElement(
                    is_categorical=True,
                    code="dataset",
                    sql_type="text",
                    label="Dataset",
                    min=None,
                    max=None,
                    enumerations={"dummy_tbi": "Dummy TBI"},
                ),
                CommonDataElement(
                    is_categorical=True,
                    code="gender_type",
                    sql_type="text",
                    label="Gender",
                    min=None,
                    max=None,
                    enumerations={"M": "Male", "F": "Female"},
                ),
            ],
            groups=[],
            longitudinal=True,
        )
        == data_models_metadata["tbi:0.1"]
    )
    assert (
        DataModelMetadata(
            code="dementia",
            version="0.1",
            label="dementia",
            variables=[
                CommonDataElement(
                    is_categorical=True,
                    code="dataset",
                    sql_type="text",
                    label="Dataset",
                    min=None,
                    max=None,
                    enumerations={
                        "ppmi": "PPMI",
                        "edsd": "EDSD",
                        "desd-synthdata": "DESD-synthdata",
                    },
                ),
                CommonDataElement(
                    is_categorical=True,
                    code="alzheimerbroadcategory",
                    sql_type="text",
                    label="Alzheimer Broad Category",
                    min=None,
                    max=None,
                    enumerations={
                        "AD": "Alzheimer’s disease",
                        "CN": "Cognitively Normal",
                        "Other": "Other",
                        "MCI": "Mild cognitive impairment",
                    },
                ),
                CommonDataElement(
                    is_categorical=False,
                    code="minimentalstate",
                    sql_type="int",
                    label="MMSE Total scores",
                    min=0.0,
                    max=30.0,
                    enumerations=None,
                ),
            ],
            groups=[],
            longitudinal=False,
        )
        == data_models_metadata["dementia:0.1"]
    )


def test_empty_initialization():
    dmr = DataModelRegistry()
    assert not dmr.data_models_metadata
    assert not dmr.datasets_locations

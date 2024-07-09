import pytest

# TODO the testing should be better once the datasets are properly distributed and the are no duplicates.
from exareme2.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    DataModelRegistry,
)
from exareme2.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    DataModelsAttributes,
)
from exareme2.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    DataModelsCDES,
)
from exareme2.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    DatasetLocation,
)
from exareme2.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    DatasetsLocations,
)
from exareme2.worker_communication import CommonDataElement
from exareme2.worker_communication import CommonDataElements
from exareme2.worker_communication import DataModelAttributes


@pytest.fixture
def mocked_datasets_locations():
    yield DatasetsLocations(
        datasets_locations={
            "tbi:0.1": {
                "dummy_tbi0": DatasetLocation(
                    worker_id="localworker1", csv_path="/opt/data/dummy_tbi0.csv"
                ),
                "dummy_tbi1": DatasetLocation(
                    worker_id="localworker2", csv_path="/opt/data/dummy_tbi1.csv"
                ),
                "dummy_tbi3": DatasetLocation(
                    worker_id="localworker2", csv_path="/opt/data/dummy_tbi3.csv"
                ),
            },
            "dementia:0.1": {
                "ppmi0": DatasetLocation(
                    worker_id="localworker1", csv_path="/opt/data/ppmi0.csv"
                ),
                "ppmi1": DatasetLocation(
                    worker_id="localworker2", csv_path="/opt/data/ppmi1.csv"
                ),
                "ppmi3": DatasetLocation(
                    worker_id="localworker2", csv_path="/opt/data/ppmi3.csv"
                ),
                "edsd0": DatasetLocation(
                    worker_id="localworker1", csv_path="/opt/data/edsd0.csv"
                ),
                "edsd1": DatasetLocation(
                    worker_id="localworker2", csv_path="/opt/data/edsd1.csv"
                ),
                "edsd3": DatasetLocation(
                    worker_id="localworker2", csv_path="/opt/data/edsd3.csv"
                ),
                "synthdata0": DatasetLocation(
                    worker_id="localworker1", csv_path="/opt/data/synthdata0.csv"
                ),
                "synthdata1": DatasetLocation(
                    worker_id="localworker2", csv_path="/opt/data/synthdata1.csv"
                ),
                "synthdata2": DatasetLocation(
                    worker_id="localworker2", csv_path="/opt/data/synthdata2.csv"
                ),
            },
        }
    )


@pytest.fixture
def mocked_data_models_cdes():
    yield DataModelsCDES(
        data_models_cdes={
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
                    "minimentalstate": CommonDataElement(
                        code="minimentalstate",
                        label="MMSE Total scores",
                        sql_type="int",
                        is_categorical=False,
                        min=0,
                        max=30,
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
                }
            ),
        }
    )


@pytest.fixture
def mocked_data_models_attributes():
    yield DataModelsAttributes(
        data_models_attributes={
            "dementia:0.1": DataModelAttributes(
                tags=["dementia"], properties={"key": ["value"]}
            ),
            "tbi:0.1": DataModelAttributes(
                tags=["tbi", "longitudinal"], properties={"key": ["value"]}
            ),
        }
    )


@pytest.fixture
def mocked_data_model_registry(
    mocked_data_models_cdes, mocked_datasets_locations, mocked_data_models_attributes
):
    data_model_registry = DataModelRegistry(
        data_models_cdes=mocked_data_models_cdes,
        datasets_locations=mocked_datasets_locations,
        data_models_attributes=mocked_data_models_attributes,
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


def test_get_data_models_attributes(mocked_data_model_registry):
    data_models_attributes = mocked_data_model_registry.get_data_models_attributes()
    assert "tbi:0.1" in data_models_attributes
    assert "dementia:0.1" in data_models_attributes
    assert (
        DataModelAttributes(tags=["tbi", "longitudinal"], properties={"key": ["value"]})
        == data_models_attributes["tbi:0.1"]
    )
    assert (
        DataModelAttributes(tags=["dementia"], properties={"key": ["value"]})
        == data_models_attributes["dementia:0.1"]
    )


def test_empty_initialization():
    dmr = DataModelRegistry()
    assert not dmr.data_models_cdes.data_models_cdes
    assert not dmr.datasets_locations.datasets_locations

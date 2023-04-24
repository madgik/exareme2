import pytest

from mipengine import AlgorithmNamesMismatchError
from mipengine import _check_algo_naming_matching


@pytest.fixture
def mismatching_algo_classes_algo_data_loaders():
    algo_classes = {
        "algo1": "algo1_class",
        "algo2": "algo2_class",
        "algo3": "algo3_class",
    }
    algo_data_loaders = {
        "algo1": "algo1_data_loader",
        "algo_name_mismath": "algo2_data_loader",
        "algo3": "algo3_data_loader",
    }
    return (algo_classes, algo_data_loaders)


@pytest.fixture
def matching_algo_classes_algo_data_loaders():
    algo_classes = {
        "algo1": "algo1_class",
        "algo2": "algo2_class",
        "algo3": "algo3_class",
    }
    algo_data_loaders = {
        "algo1": "algo1_data_loader",
        "algo2": "algo2_data_loader",
        "algo3": "algo3_data_loader",
    }
    return (algo_classes, algo_data_loaders)


def test_check_algo_naming_mismatching(
    mismatching_algo_classes_algo_data_loaders, matching_algo_classes_algo_data_loaders
):
    with pytest.raises(AlgorithmNamesMismatchError) as exc:
        _check_algo_naming_matching(
            mismatching_algo_classes_algo_data_loaders[0],
            mismatching_algo_classes_algo_data_loaders[1],
        )

    _check_algo_naming_matching(
        matching_algo_classes_algo_data_loaders[0],
        matching_algo_classes_algo_data_loaders[1],
    )
    assert True

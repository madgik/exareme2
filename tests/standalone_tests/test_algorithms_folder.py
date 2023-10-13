import importlib
import os

import pytest

import exareme2
from tests import TEST_ALGORITHMS_FOLDER


@pytest.fixture
def set_default_algorithms_folder():
    if exareme2.ALGORITHM_FOLDERS_ENV_VARIABLE in os.environ:
        del os.environ[exareme2.ALGORITHM_FOLDERS_ENV_VARIABLE]


def test_default_algorithms_folder(set_default_algorithms_folder):
    assert (
        exareme2.ALGORITHM_FOLDERS
        == "./exareme2/algorithms/in_database,./exareme2/algorithms/native_python"
    )


@pytest.fixture
def set_test_algorithms_folder():
    os.environ[exareme2.ALGORITHM_FOLDERS_ENV_VARIABLE] = TEST_ALGORITHMS_FOLDER
    yield
    del os.environ[exareme2.ALGORITHM_FOLDERS_ENV_VARIABLE]


@pytest.mark.slow
def test_test_algorithms_folder(set_test_algorithms_folder):
    importlib.reload(exareme2)
    assert exareme2.ALGORITHM_FOLDERS == "./tests/algorithms"

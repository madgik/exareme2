import importlib
import os

import pytest

from tests import TEST_ALGORITHMS_FOLDER


@pytest.fixture
def set_default_algorithms_folder():
    import mipengine

    if mipengine.ALGORITHM_FOLDERS_ENV_VARIABLE in os.environ:
        del os.environ[mipengine.ALGORITHM_FOLDERS_ENV_VARIABLE]


def test_default_algorithms_folder(set_default_algorithms_folder):
    import mipengine

    assert mipengine.ALGORITHM_FOLDERS == "./mipengine/algorithms"


@pytest.fixture
def set_test_algorithms_folder():
    import mipengine

    os.environ[mipengine.ALGORITHM_FOLDERS_ENV_VARIABLE] = TEST_ALGORITHMS_FOLDER
    yield
    del os.environ[mipengine.ALGORITHM_FOLDERS_ENV_VARIABLE]


def test_test_algorithms_folder(set_test_algorithms_folder):
    import mipengine

    importlib.reload(mipengine)
    assert mipengine.ALGORITHM_FOLDERS == "./tests/algorithms"

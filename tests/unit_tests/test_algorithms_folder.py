import importlib
import os

import pytest

import mipengine
from tests import TEST_ALGORITHMS_FOLDER


@pytest.fixture
def set_default_algorithms_folder():
    if mipengine.ALGORITHMS_FOLDER_ENV_VARIABLE in os.environ:
        del os.environ[mipengine.ALGORITHMS_FOLDER_ENV_VARIABLE]


def test_default_algorithms_folder(set_default_algorithms_folder):
    importlib.reload(mipengine)
    assert mipengine.ALGORITHMS_FOLDER == "mipengine.algorithms"


@pytest.fixture
def set_test_algorithms_folder():
    os.environ[mipengine.ALGORITHMS_FOLDER_ENV_VARIABLE] = TEST_ALGORITHMS_FOLDER
    yield
    del os.environ[mipengine.ALGORITHMS_FOLDER_ENV_VARIABLE]


def test_test_algorithms_folder(set_test_algorithms_folder):
    importlib.reload(mipengine)
    assert mipengine.ALGORITHMS_FOLDER == "tests.algorithms"

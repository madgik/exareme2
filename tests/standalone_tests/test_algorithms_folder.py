# import importlib
# import os
#
# import pytest
#
# import exareme2
# from tests import TEST_ALGORITHMS_FOLDER
#
#
# @pytest.fixture
# def set_default_algorithms_folder():
#     if exareme2.EXAREME2_ALGORITHM_FOLDERS_ENV_VARIABLE in os.environ:
#         del os.environ[exareme2.EXAREME2_ALGORITHM_FOLDERS_ENV_VARIABLE]
#
#
# def test_default_algorithms_folder(set_default_algorithms_folder):
#     assert exareme2.EXAREME2_ALGORITHM_FOLDERS == "./exareme2/algorithms/exareme2"
#
#
# @pytest.fixture
# def set_test_algorithms_folder():
#     os.environ[exareme2.EXAREME2_ALGORITHM_FOLDERS_ENV_VARIABLE] = (
#         TEST_ALGORITHMS_FOLDER
#     )
#     yield
#     del os.environ[exareme2.EXAREME2_ALGORITHM_FOLDERS_ENV_VARIABLE]
#
#
# @pytest.mark.slow
# def test_test_algorithms_folder(set_test_algorithms_folder):
#     importlib.reload(exareme2)
#     assert exareme2.EXAREME2_ALGORITHM_FOLDERS == "./tests/algorithms"

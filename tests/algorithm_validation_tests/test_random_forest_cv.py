from pathlib import Path

import numpy as np
import pytest

from tests.algorithm_validation_tests.helpers import algorithm_request
from tests.algorithm_validation_tests.helpers import get_test_params
from tests.algorithm_validation_tests.helpers import parse_response

algorithm_name = "random_forest_cv"

expected_file = Path(__file__).parent / "expected" / f"{algorithm_name}_expected.json"


@pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
def test_random_forest_cv(test_input, expected):
    response = algorithm_request(algorithm_name, test_input)
    result = parse_response(response)

    # sort records by variable and dataset in order to compare them

    expected_accuracy = np.array(expected["accuracy_list"])
    result_accuracy = np.array(result["accuracy_list"])

    mean_expected_accuracy = np.mean(expected_accuracy)
    mean_result_accuracy = np.mean(result_accuracy)

    assert 0 <= mean_result_accuracy <= 1
    assert 0 <= mean_expected_accuracy <= 1
    assert abs(mean_result_accuracy - mean_expected_accuracy) <= 0.2

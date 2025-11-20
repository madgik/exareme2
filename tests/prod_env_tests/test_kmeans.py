import json
from pathlib import Path

import pytest

from tests.algorithm_validation_tests.exaflow.conftest import algorithm_request
from tests.algorithm_validation_tests.exaflow.helpers import get_test_params

expected_file = Path(__file__).parent / "expected" / "kmeans_expected.json"


@pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
def test_kmeans(test_input, expected):
    response = algorithm_request("kmeans", test_input)
    try:
        result = json.loads(response.text)
    except json.decoder.JSONDecodeError:
        raise ValueError(f"The result is not valid json:\n{response.text}") from None

    # this test only ensures that the algorithm runs smoothly without errors
    assert result

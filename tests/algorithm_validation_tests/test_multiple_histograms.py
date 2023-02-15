import json
from pathlib import Path

import pytest

from tests.algorithm_validation_tests.helpers import algorithm_request
from tests.algorithm_validation_tests.helpers import get_test_params

algorithm_name = "multiple_histograms"

expected_file = Path(__file__).parent / "expected" / f"{algorithm_name}_expected.json"


@pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
def test_histogram(test_input, expected):
    response = algorithm_request(algorithm_name, test_input)
    try:
        result = json.loads(response.text)
    except json.decoder.JSONDecodeError:
        raise ValueError(f"The result is not valid json:\n{response.text}") from None

    # sort records by variable and dataset in order to compare them

    assert expected["histogram"] == result["histogram"]

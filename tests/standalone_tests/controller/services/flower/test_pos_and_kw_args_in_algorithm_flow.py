import json

import pytest
import requests

from tests.standalone_tests.conftest import ALGORITHMS_URL


@pytest.mark.slow
def test_pos_and_kw_args_in_algorithm_flow(
    globalworker_worker_service,
    localworker1_worker_service,
    load_data_localworker1,
    load_test_data_globalworker,
    controller_service_with_localworker1,
):
    algorithm_name = "logistic_regression_fedaverage_flower"
    request_dict = {
        "inputdata": {
            "y": ["gender"],
            "x": ["lefthippocampus"],
            "data_model": "dementia:0.1",
            "datasets": [
                "ppmi0",
                "ppmi1",
                "ppmi2",
                "ppmi3",
            ],
            "validation_datasets": ["ppmi_test"],
            "filters": None,
        },
    }

    algorithm_url = ALGORITHMS_URL + "/" + algorithm_name

    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    response = requests.post(
        algorithm_url,
        data=json.dumps(request_dict),
        headers=headers,
    )
    assert response.status_code == 200

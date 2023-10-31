import json

import pytest
import requests

from tests.standalone_tests.conftest import SMPC_ALGORITHMS_URL
from tests.standalone_tests.conftest import kill_service
from tests.standalone_tests.conftest import smpc_controller_service_with_dp


@pytest.fixture
def algorithm_request():
    algorithm_name = "pca"
    request_dict = {
        "inputdata": {
            "data_model": "dementia:0.1",
            "datasets": [
                "ppmi0",
                "ppmi1",
                "ppmi2",
                "ppmi3",
                "ppmi4",
                "ppmi5",
                "ppmi6",
                "ppmi7",
            ],
            "y": ["leftamygdala", "lefthippocampus"],
        },
    }
    return (algorithm_name, request_dict)


@pytest.mark.slow
@pytest.mark.very_slow
@pytest.mark.smpc
@pytest.mark.smpc_cluster
def test_dp_vs_nondp_algorithm_execution(
    smpc_cluster,
    smpc_globalnode_node_service,
    smpc_localnode1_node_service,
    load_data_smpc_localnode1,
    smpc_localnode2_node_service,
    load_data_smpc_localnode2,
    smpc_controller_service,
    algorithm_request,
):
    # This test executes the same algorithm twice, first with the differential
    # privacy disabled and then with the differential privacy feature enabled
    # and with a high sensitivity value. This will (most likely) cause the
    # result of the two executions of the algorithm to vary...

    algorithm_url = SMPC_ALGORITHMS_URL + "/" + algorithm_request[0]
    algorithm_input = algorithm_request[1]

    headers = {"Content-type": "application/json", "Accept": "text/plain"}

    # smpc_controller_service has dp disabled
    response_without_dp = requests.post(
        algorithm_url,
        data=json.dumps(algorithm_input),
        headers=headers,
    )
    kill_service(smpc_controller_service)

    # smpc_controller_service_with_dp starts a controller with dp enabled
    smpc_dp_controller_service = smpc_controller_service_with_dp()
    response_with_dp = requests.post(
        algorithm_url,
        data=json.dumps(algorithm_input),
        headers=headers,
    )
    kill_service(smpc_dp_controller_service)

    assert response_without_dp.status_code == 200
    assert response_with_dp.status_code == 200
    assert response_without_dp.content != response_with_dp.content

import requests

from mipengine.node_exceptions import SMPCCommunicationError

ADD_DATASET_ENDPOINT = "/api/update-dataset/"
TRIGGER_COMPUTATION_ENDPOINT = "/api/secure-aggregation/job-id/"
GET_RESULT_ENDPOINT = "/api/get-result/job-id/"


def load_data_to_smpc_client(client_address, jobid, values):
    request_url = client_address + ADD_DATASET_ENDPOINT + jobid
    request_headers = {"Content-type": "application/json", "Accept": "text/plain"}
    response = requests.post(
        request_url,
        data=values,
        headers=request_headers,
    )
    if response.status_code != 200:
        raise SMPCCommunicationError(
            f"Response status code: {response.status_code} \n Body:{response.text}"
        )


def get_smpc_result(coordinator_address, jobid):
    request_url = coordinator_address + GET_RESULT_ENDPOINT + jobid
    request_headers = {"Content-type": "application/json", "Accept": "text/plain"}
    response = requests.get(
        url=request_url,
        headers=request_headers,
    )
    if response.status_code != 200:
        raise SMPCCommunicationError(
            f"Response status code: {response.status_code} \n Body:{response.text}"
        )
    return response

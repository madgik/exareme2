import json
from logging import Logger
from typing import List

import requests

from exareme2.smpc_DTOs import DifferentialPrivacyParams
from exareme2.smpc_DTOs import SMPCRequestData
from exareme2.smpc_DTOs import SMPCRequestType

ADD_DATASET_ENDPOINT = "/api/update-dataset/"
TRIGGER_COMPUTATION_ENDPOINT = "/api/secure-aggregation/job-id/"
GET_RESULT_ENDPOINT = "/api/get-result/job-id/"


def _get_smpc_load_data_request_data_structure(data_values: str):
    """
    The current approach with the SMPC cluster is to send all computations as floats.
    That way we don't need to have separate operations for sum-int and sum-float.
    """
    data = {"type": "float", "data": json.loads(data_values)}
    return json.dumps(data)


def load_data_to_smpc_client(client_address: str, jobid: str, values: str):
    request_url = client_address + ADD_DATASET_ENDPOINT + jobid
    request_headers = {"Content-type": "application/json", "Accept": "text/plain"}
    response = requests.post(
        url=request_url,
        data=_get_smpc_load_data_request_data_structure(values),
        headers=request_headers,
    )
    if response.status_code != 200:
        raise SMPCCommunicationError(
            f"Response status code: {response.status_code} \n Body:{response.text}"
        )


def get_smpc_result(coordinator_address: str, jobid: str) -> str:
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
    return response.text


def trigger_smpc(
    logger: Logger,
    coordinator_address: str,
    jobid: str,
    payload: SMPCRequestData,
):
    request_url = coordinator_address + TRIGGER_COMPUTATION_ENDPOINT + jobid
    request_headers = {"Content-type": "application/json", "Accept": "text/plain"}
    logger.info(f"Starting SMPC with {jobid=}...")
    logger.debug(f"{request_url=}")
    logger.debug(f"{payload=}")
    response = requests.post(
        url=request_url,
        data=payload,
        headers=request_headers,
    )
    if response.status_code != 200:
        raise SMPCCommunicationError(
            f"Response status code: {response.status_code} \n Body:{response.text}"
        )


def create_payload(
    computation_type: SMPCRequestType,
    clients: List[str],
    dp_params: DifferentialPrivacyParams = None,
) -> SMPCRequestData:
    if dp_params:
        return SMPCRequestData(
            computationType=computation_type,
            clients=clients,
            c=dp_params.sensitivity,
            e=dp_params.privacy_budget,
        ).json()
    else:
        return SMPCRequestData(computationType=computation_type, clients=clients).json()


def trigger_dp(
    logger: Logger,
    coordinator_address: str,
    jobid: str,
    computation_type: SMPCRequestType,
    clients: List[str],
):
    #     http://{{coordinator}}:{{coordinator-port}}/api/secure-aggregation/job-id/testKey13

    # {
    #     "computationType": "sum",
    #     "returnUrl": "http://localhost:4100",
    #     "clients": ["ZuellingPharma"],
    #     "dp": {
    #         "c": 1,
    #         "e": 1
    #     }
    # }
    pass


def validate_smpc_usage(use_smpc: bool, smpc_enabled: bool, smpc_optional: bool):
    """
    Validates if smpc can be used or if it must be used based on the configs.
    """
    if use_smpc and not smpc_enabled:
        raise SMPCUsageError("SMPC cannot be used, since it's not enabled.")

    if not use_smpc and smpc_enabled and not smpc_optional:
        raise SMPCUsageError(
            "The computation cannot be made without SMPC. SMPC usage is not optional."
        )


class SMPCUsageError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class SMPCCommunicationError(Exception):
    pass


class SMPCComputationError(Exception):
    pass

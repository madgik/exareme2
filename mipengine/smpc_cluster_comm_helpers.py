from logging import Logger
from typing import List

import requests

from mipengine.smpc_DTOs import SMPCRequestData
from mipengine.smpc_DTOs import SMPCRequestType

ADD_DATASET_ENDPOINT = "/api/update-dataset/"
TRIGGER_COMPUTATION_ENDPOINT = "/api/secure-aggregation/job-id/"
GET_RESULT_ENDPOINT = "/api/get-result/job-id/"


def load_data_to_smpc_client(client_address: str, jobid: str, values: str):
    request_url = client_address + ADD_DATASET_ENDPOINT + jobid
    request_headers = {"Content-type": "application/json", "Accept": "text/plain"}
    response = requests.post(
        url=request_url,
        data=values,
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
    computation_type: SMPCRequestType,
    clients: List[int],
):
    request_url = coordinator_address + TRIGGER_COMPUTATION_ENDPOINT + jobid
    request_headers = {"Content-type": "application/json", "Accept": "text/plain"}
    data = SMPCRequestData(computationType=computation_type, clients=clients).json()
    logger.info(f"Starting SMPC with {jobid=}...")
    logger.debug(f"{request_url=}")
    logger.debug(f"{data=}")
    response = requests.post(
        url=request_url,
        data=data,
        headers=request_headers,
    )
    if response.status_code != 200:
        raise SMPCCommunicationError(
            f"Response status code: {response.status_code} \n Body:{response.text}"
        )


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

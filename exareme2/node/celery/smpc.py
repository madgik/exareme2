from typing import Optional

from celery import shared_task

import exareme2.node.services.in_database.smpc as smpc_service


@shared_task
def validate_smpc_templates_match(
    request_id: str,
    table_name: str,
):
    smpc_service.validate_smpc_templates_match(request_id, table_name)


@shared_task
def load_data_to_smpc_client(request_id: str, table_name: str, jobid: str) -> str:
    return smpc_service.load_data_to_smpc_client(request_id, table_name, jobid)


@shared_task
def get_smpc_result(
    request_id: str,
    jobid: str,
    context_id: str,
    command_id: str,
    command_subid: Optional[str] = "0",
) -> str:
    return smpc_service.get_smpc_result(
        request_id, jobid, context_id, command_id, command_subid
    ).json()

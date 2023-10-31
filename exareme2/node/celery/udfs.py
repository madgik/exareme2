from typing import Optional

from celery import shared_task

import exareme2.node.services.in_database.udfs as udfs_service
from exareme2.node_communication import NodeUDFKeyArguments
from exareme2.node_communication import NodeUDFPosArguments


@shared_task
def run_udf(
    request_id: str,
    command_id: str,
    context_id: str,
    func_name: str,
    positional_args_json: str,
    keyword_args_json: str,
    use_smpc: bool = False,
    output_schema: Optional[str] = None,
) -> str:
    positional_args = NodeUDFPosArguments.parse_raw(positional_args_json)
    keyword_args = NodeUDFKeyArguments.parse_raw(keyword_args_json)
    return udfs_service.run_udf(
        request_id,
        command_id,
        context_id,
        func_name,
        positional_args,
        keyword_args,
        use_smpc,
        output_schema,
    ).json()

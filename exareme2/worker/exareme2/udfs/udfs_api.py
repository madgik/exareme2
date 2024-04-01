from typing import Optional

from celery import shared_task

from exareme2.worker.exareme2.udfs import udfs_service
from exareme2.worker_communication import WorkerUDFKeyArguments
from exareme2.worker_communication import WorkerUDFPosArguments


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
    positional_args = WorkerUDFPosArguments.parse_raw(positional_args_json)
    keyword_args = WorkerUDFKeyArguments.parse_raw(keyword_args_json)
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

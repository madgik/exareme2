from celery import shared_task

from exareme2.worker.exaflow.yesql import yesql_service


@shared_task
def run_yesql(
    request_id,
    udf_registry_key: str,
    params: dict,
):
    return yesql_service.run_yesql(request_id, udf_registry_key, params)

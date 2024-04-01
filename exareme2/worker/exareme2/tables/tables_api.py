from typing import List

from celery import shared_task

from exareme2.worker.exareme2.tables import tables_service
from exareme2.worker_communication import TableInfo
from exareme2.worker_communication import TableSchema


@shared_task
def get_tables(request_id: str, context_id: str) -> List[str]:
    return tables_service.get_tables(request_id, context_id)


@shared_task
def get_remote_tables(request_id: str, context_id: str) -> List[str]:
    return tables_service.get_remote_tables(request_id, context_id)


@shared_task
def get_merge_tables(request_id: str, context_id: str) -> List[str]:
    return tables_service.get_merge_tables(request_id, context_id)


@shared_task
def create_table(
    request_id: str, context_id: str, command_id: str, schema_json: str
) -> str:
    table_schema = TableSchema.parse_raw(schema_json)
    return tables_service.create_table(
        request_id, context_id, command_id, table_schema
    ).json()


@shared_task
def create_remote_table(
    request_id: str,
    table_name: str,
    table_schema_json: str,
    monetdb_socket_address: str,
):
    table_schema = TableSchema.parse_raw(table_schema_json)
    tables_service.create_remote_table(
        request_id, table_name, table_schema, monetdb_socket_address
    )


@shared_task
def create_merge_table(
    request_id: str, context_id: str, command_id: str, table_infos_json: List[str]
) -> str:
    table_infos = [
        TableInfo.parse_raw(table_info_json) for table_info_json in table_infos_json
    ]
    return tables_service.create_merge_table(
        request_id, context_id, command_id, table_infos
    ).json()


@shared_task
def get_table_data(request_id: str, table_name: str) -> str:
    return tables_service.get_table_data(request_id, table_name).json()

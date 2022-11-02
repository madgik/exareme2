from typing import List

import pymonetdb

from mipengine.exceptions import IncompatibleSchemasMergeException
from mipengine.exceptions import TablesNotFound
from mipengine.node.monetdb_interface.common_actions import (
    convert_schema_to_sql_query_format,
)
from mipengine.node.monetdb_interface.common_actions import get_table_names
from mipengine.node.monetdb_interface.guard import is_list_of_identifiers
from mipengine.node.monetdb_interface.guard import is_valid_table_schema
from mipengine.node.monetdb_interface.guard import sql_injection_guard
from mipengine.node.monetdb_interface.monet_db_facade import db_execute
from mipengine.node_tasks_DTOs import TableSchema
from mipengine.node_tasks_DTOs import TableType


def get_merge_tables_names(context_id: str) -> List[str]:
    return get_table_names(TableType.MERGE, context_id)


@sql_injection_guard(table_name=str.isidentifier, table_schema=is_valid_table_schema)
def create_merge_table(table_name: str, table_schema: TableSchema):
    columns_schema = convert_schema_to_sql_query_format(table_schema)
    db_execute(f"CREATE MERGE TABLE {table_name} ( {columns_schema} )")


@sql_injection_guard(
    merge_table_name=str.isidentifier,
    table_names=is_list_of_identifiers,
)
def add_to_merge_table(merge_table_name: str, table_names: List[str]):
    for name in table_names:
        try:
            db_execute(f"ALTER TABLE {merge_table_name} ADD TABLE {name.lower()}")
        except pymonetdb.exceptions.OperationalError as exc:
            if str(exc).startswith("3F000"):
                raise IncompatibleSchemasMergeException(table_names)
            if str(exc).startswith("42S02"):
                raise TablesNotFound([name])
            else:
                raise exc

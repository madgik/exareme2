from typing import List

import pymonetdb

from exareme2.exceptions import IncompatibleSchemasMergeException
from exareme2.exceptions import TablesNotFound
from exareme2.node.monetdb_interface.common_actions import (
    convert_schema_to_sql_query_format,
)
from exareme2.node.monetdb_interface.common_actions import get_table_names
from exareme2.node.monetdb_interface.guard import is_list_of_identifiers
from exareme2.node.monetdb_interface.guard import is_valid_table_schema
from exareme2.node.monetdb_interface.guard import sql_injection_guard
from exareme2.node.monetdb_interface.monet_db_facade import db_execute_query
from exareme2.node_tasks_DTOs import TableSchema
from exareme2.node_tasks_DTOs import TableType


def get_merge_tables_names(context_id: str) -> List[str]:
    return get_table_names(TableType.MERGE, context_id)


@sql_injection_guard(
    table_name=str.isidentifier,
    table_schema=is_valid_table_schema,
    merge_table_names=is_list_of_identifiers,
)
def create_merge_table(
    table_name: str,
    table_schema: TableSchema,
    merge_table_names: List[str],
):
    """
    The schema of the 1st table is used as the merge table schema.
    If there is an incompatibility or a table doesn't exist the db will throw an error.
    """
    columns_schema = convert_schema_to_sql_query_format(table_schema)
    merge_table_query = f"CREATE MERGE TABLE {table_name} ( {columns_schema} ); "
    for name in merge_table_names:
        merge_table_query += f"ALTER TABLE {table_name} ADD TABLE {name.lower()}; "

    try:
        db_execute_query(merge_table_query)
    except pymonetdb.exceptions.ProgrammingError or pymonetdb.exceptions.OperationalError as exc:
        if str(exc).startswith("3F000"):
            raise IncompatibleSchemasMergeException(merge_table_names)
        if str(exc).startswith("42S02"):
            raise TablesNotFound(merge_table_names)
        else:
            raise exc

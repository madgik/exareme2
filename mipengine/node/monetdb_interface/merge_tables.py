from typing import List

import pymonetdb

from mipengine.node_exceptions import IncompatibleSchemasMergeException
from mipengine.node_exceptions import IncompatibleTableTypes
from mipengine.node_exceptions import TablesNotFound
from mipengine.node.monetdb_interface.common_actions import (
    convert_schema_to_sql_query_format,
)
from mipengine.node.monetdb_interface.common_actions import get_table_names
from mipengine.node.monetdb_interface.monet_db_connection import MonetDB
from mipengine.node_tasks_DTOs import TableSchema
from mipengine.node_tasks_DTOs import TableType


def get_merge_tables_names(context_id: str) -> List[str]:
    return get_table_names(TableType.MERGE, context_id)


def create_merge_table(table_name: str, table_schema: TableSchema):
    columns_schema = convert_schema_to_sql_query_format(table_schema)
    MonetDB().execute(f"CREATE MERGE TABLE {table_name} ( {columns_schema} )")


def get_non_existing_tables(table_names: List[str]) -> List[str]:
    names_clause = str(table_names)[1:-1]
    existing_tables = MonetDB().execute_and_fetchall(
        f"SELECT name FROM tables WHERE name IN ({names_clause})"
    )
    existing_table_names = [table[0] for table in existing_tables]
    return [name for name in table_names if name not in existing_table_names]


def add_to_merge_table(merge_table_name: str, table_names: List[str]):
    non_existing_tables = get_non_existing_tables(table_names)

    try:
        for name in table_names:
            MonetDB().execute(
                f"ALTER TABLE {merge_table_name} ADD TABLE {name.lower()}"
            )

    except pymonetdb.exceptions.OperationalError as exc:
        if str(exc).startswith("3F000"):
            raise IncompatibleSchemasMergeException(table_names)
        elif str(exc).startswith("42S02"):
            raise TablesNotFound(non_existing_tables)
        else:
            raise exc


def validate_tables_can_be_merged(tables_names: List[str]):
    table_names = ",".join(f"'{table}'" for table in tables_names)

    distinct_table_types = MonetDB().execute_and_fetchall(
        f"""
        SELECT DISTINCT(type)
        FROM tables
        WHERE
        system = false
        AND
        name in ({table_names})"""
    )

    if len(distinct_table_types) != 1:
        raise IncompatibleTableTypes(distinct_table_types)

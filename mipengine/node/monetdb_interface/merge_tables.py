from typing import List

import pymonetdb

from mipengine.node.monetdb_interface.common_actions import get_table_schema
from mipengine.node.monetdb_interface.common_actions import get_table_type
from mipengine.node_exceptions import IncompatibleSchemasMergeException
from mipengine.node_exceptions import IncompatibleTableTypes
from mipengine.node_exceptions import TablesNotFound
from mipengine.node_tasks_DTOs import TableInfo
from mipengine.node.monetdb_interface.common_actions import (
    convert_schema_to_sql_query_format,
)
from mipengine.node.monetdb_interface.common_actions import get_table_names
from mipengine.node.monetdb_interface.monet_db_connection import MonetDB
from mipengine.node_tasks_DTOs import TableType


def get_merge_tables_names(context_id: str) -> List[str]:
    return get_table_names(TableType.MERGE, context_id)


def create_merge_table(table_info: TableInfo):
    columns_schema = convert_schema_to_sql_query_format(table_info.schema_)
    MonetDB().execute(f"CREATE MERGE TABLE {table_info.name} ( {columns_schema} )")


def add_to_merge_table(merge_table_name: str, table_names: List[str]):
    try:
        for name in table_names:
            MonetDB().execute(
                f"ALTER TABLE {merge_table_name} ADD TABLE {name.lower()}"
            )

    except pymonetdb.exceptions.OperationalError as exc:
        if str(exc).startswith("3F000"):
            table_infos = [
                TableInfo(
                    name=name,
                    schema_=get_table_schema(name),
                    type_=get_table_type(name),
                )
                for name in table_names
            ]
            raise IncompatibleSchemasMergeException(table_infos)
        else:
            raise exc


def validate_tables_can_be_merged(table_names: List[str]):
    names_clause = ",".join(f"'{table}'" for table in table_names)
    existing_table_names_and_types = MonetDB().execute_and_fetchall(
        f"SELECT name, type FROM tables WHERE name IN ({names_clause})"
    )
    existing_table_names, existing_table_types = zip(*existing_table_names_and_types)
    non_existing_tables = [
        name for name in table_names if name not in existing_table_names
    ]
    if non_existing_tables:
        raise TablesNotFound(non_existing_tables)
    distinct_table_types = set(existing_table_types)
    if len(distinct_table_types) != 1:
        raise IncompatibleTableTypes(distinct_table_types)

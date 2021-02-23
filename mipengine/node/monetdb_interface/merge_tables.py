from typing import List

import pymonetdb

from mipengine.node.monetdb_interface import common
from mipengine.node.monetdb_interface import tables
from mipengine.node.monetdb_interface.common import connection
from mipengine.node.monetdb_interface.common import convert_schema_to_sql_query_format
from mipengine.node.monetdb_interface.common import cursor
from mipengine.node.tasks.data_classes import TableInfo
from mipengine.utils.custom_exception import IncompatibleSchemasMergeException
from mipengine.utils.custom_exception import TableCannotBeFound


def get_merge_tables_names(context_id: str) -> List[str]:
    return common.get_tables_names("merge", context_id)


def create_merge_table(table_info: TableInfo):
    columns_schema = convert_schema_to_sql_query_format(table_info.schema)
    cursor.execute(f"CREATE MERGE TABLE {table_info.name} ( {columns_schema} )")


def get_non_existing_tables(table_names: List[str]) -> List[str]:
    names_clause = str(table_names)[1:-1]
    cursor.execute(f"SELECT name FROM tables WHERE name IN({names_clause})")
    existing_table_names = [table[0] for table in cursor]
    return [name for name in table_names if name not in existing_table_names]


def add_to_merge_table(merge_table_name: str, partition_tables_names: List[str]):
    non_existing_tables = get_non_existing_tables(partition_tables_names)
    table_infos = [TableInfo(name, tables.get_table_schema(name)) for name in partition_tables_names]

    try:
        for name in partition_tables_names:
            cursor.execute(f"ALTER TABLE {merge_table_name} ADD TABLE {name.lower()}")

    except pymonetdb.exceptions.OperationalError as exc:
        if str(exc).startswith('3F000'):
            connection.rollback()
            raise IncompatibleSchemasMergeException(table_infos)
        elif str(exc).startswith('42S02'):
            connection.rollback()
            raise TableCannotBeFound(non_existing_tables)
        else:
            connection.rollback()
            raise exc
    connection.commit()

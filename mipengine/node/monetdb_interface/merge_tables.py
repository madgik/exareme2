from typing import List

from mipengine.utils.custom_exception import IncompatibleSchemasMergeException, TableCannotBeFound
from mipengine.node.monetdb_interface import common, tables
from mipengine.node.monetdb_interface.common import convert_table_info_to_sql_query_format, cursor, connection
from mipengine.node.tasks.data_classes import TableInfo


def get_merge_tables_names(context_id: str) -> List[str]:
    return common.get_tables_names("merge", context_id)


def create_merge_table(table_info: TableInfo):
    columns_schema = convert_table_info_to_sql_query_format(table_info)
    cursor.execute(f"CREATE MERGE TABLE {table_info.name} ( {columns_schema} )")


def get_non_existing_tables(tables_names: List[str]) -> List[str]:
    cursor.execute(f"SELECT name FROM tables WHERE name IN({str(tables_names)[1:-1]})")
    existing_tables = [table[0] for table in cursor]
    print(existing_tables)
    print(tables_names)
    return list(list(set(tables_names)-set(existing_tables)))


def add_to_merge_table(merge_table_name: str, partition_tables_names: List[str]):
    try:
        non_existing_tables = get_non_existing_tables(partition_tables_names)
        print(non_existing_tables)
        table_infos = [TableInfo(name, tables.get_table_schema(name)) for name in partition_tables_names]
        for name in partition_tables_names:
            cursor.execute(f"ALTER TABLE {merge_table_name} ADD TABLE {name.lower()}")

    except Exception as exc:
        code = str(exc)[0:5]
        print(code)
        if code == '3F000':
            raise IncompatibleSchemasMergeException(table_infos)
        elif code == '42S02':
            raise TableCannotBeFound(non_existing_tables)
        else:
            raise exc
    connection.commit()

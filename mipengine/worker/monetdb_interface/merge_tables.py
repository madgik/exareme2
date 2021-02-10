from typing import List

from mipengine.utils.custom_exception import IncompatibleSchemasMergeException, TableCannotBeFound
from mipengine.worker.monetdb_interface import common
from mipengine.worker.monetdb_interface.common import convert_table_info_to_sql_query_format, cursor, connection
from mipengine.worker.tasks.data_classes import TableInfo


def get_merge_tables_names(context_id: str) -> List[str]:
    return common.get_tables_names("merge", context_id)


def create_merge_table(table_info: TableInfo):
    columns_schema = convert_table_info_to_sql_query_format(table_info)
    cursor.execute(f"CREATE MERGE TABLE {table_info.name} ( {columns_schema} )")
    connection.commit()


def add_to_merge_table(merge_table_name: str, partition_tables_names: List[str]):
    try:
        for table in partition_tables_names:
            current_table = table
            cursor.execute(f"ALTER TABLE {merge_table_name} ADD TABLE {table.lower()}")

    except Exception as exc:
        code = str(exc)[0:5]
        if code == '3F000':
            raise IncompatibleSchemasMergeException(current_table)
        elif code == '42S02':
            raise TableCannotBeFound(current_table)
        else:
            raise exc
        cursor.execute(f"DROP TABLE {merge_table_name}")
        connection.commit()

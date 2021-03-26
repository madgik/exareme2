from typing import List

from mipengine.common.validate_identifier_names import validate_identifier_names
from mipengine.node.monetdb_interface import common_actions
from mipengine.node.monetdb_interface.common_actions import convert_schema_to_sql_query_format
from mipengine.common.node_tasks_DTOs import TableInfo
from mipengine.node.monetdb_interface.monet_db_connection import MonetDB


def get_tables_names(context_id: str) -> List[str]:
    return common_actions.get_tables_names("normal", context_id)


@validate_identifier_names
def create_table(table_info: TableInfo):
    columns_schema = convert_schema_to_sql_query_format(table_info.schema)
    MonetDB().execute(f"CREATE TABLE {table_info.name} ( {columns_schema} )")

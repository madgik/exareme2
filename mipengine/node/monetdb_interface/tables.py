from typing import List
from typing import Union

from mipengine.node.monetdb_interface import common_actions
from mipengine.node.monetdb_interface.common_actions import (
    convert_schema_to_sql_query_format,
)
from mipengine.node.monetdb_interface.monet_db_connection import MonetDB
from mipengine.node.monetdb_interface.monet_db_connection import monetdb
from mipengine.node_tasks_DTOs import TableSchema
from mipengine.node_tasks_DTOs import TableType


def get_table_names(context_id: str) -> List[str]:
    return common_actions.get_table_names(TableType.NORMAL, context_id)


def create_table(table_name: str, table_schema: TableSchema):
    columns_schema = convert_schema_to_sql_query_format(table_schema)
    monetdb.execute(f"CREATE TABLE {table_name} ( {columns_schema} )")


# TODO:Should validate the arguments, will be fixed with pydantic
def insert_data_to_table(
    table_name: str, table_values: List[List[Union[str, int, float]]]
):
    row_length = len(table_values[0])
    if all(len(row) != row_length for row in table_values):
        raise Exception("Row counts does not match")
    params_format = ", ".join(("%s",) * row_length)
    sql_clause = "INSERT INTO %s VALUES (%s)" % (table_name, params_format)
    monetdb.execute(query=sql_clause, parameters=table_values, many=True)

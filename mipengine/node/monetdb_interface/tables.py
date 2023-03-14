from itertools import chain
from typing import List
from typing import Union

from mipengine.node.monetdb_interface import common_actions
from mipengine.node.monetdb_interface.common_actions import (
    convert_schema_to_sql_query_format,
)
from mipengine.node.monetdb_interface.guard import is_valid_table_schema
from mipengine.node.monetdb_interface.guard import sql_injection_guard
from mipengine.node.monetdb_interface.monet_db_facade import db_execute_query
from mipengine.node_tasks_DTOs import TableSchema
from mipengine.node_tasks_DTOs import TableType


def get_table_names(context_id: str) -> List[str]:
    return common_actions.get_table_names(TableType.NORMAL, context_id)


@sql_injection_guard(table_name=str.isidentifier, table_schema=is_valid_table_schema)
def create_table(table_name: str, table_schema: TableSchema):
    columns_schema = convert_schema_to_sql_query_format(table_schema)
    db_execute_query(f"CREATE TABLE {table_name} ( {columns_schema} )")


@sql_injection_guard(table_name=str.isidentifier, table_values=None)
def insert_data_to_table(
    table_name: str, table_values: List[List[Union[str, int, float]]]
):
    row_length = len(table_values[0])
    if all(len(row) != row_length for row in table_values):
        raise Exception("Row counts does not match")

    # In order to achieve insertion with parameters we need to create query to the following format:
    # INSERT INTO <table_name> VALUES (%s, %s), (%s, %s);
    # The following variable 'values' create that specific str according to row_length and the amount of the rows.
    values = ", ".join(
        "(" + ", ".join("%s" for _ in range(row_length)) + ")" for _ in table_values
    )

    sql_clause = f"INSERT INTO {table_name} VALUES {values}"
    db_execute_query(query=sql_clause, parameters=list(chain(*table_values)))

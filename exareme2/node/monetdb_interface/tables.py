from typing import List
from typing import Union

from exareme2.node.monetdb_interface import common_actions
from exareme2.node.monetdb_interface.common_actions import (
    convert_schema_to_sql_query_format,
)
from exareme2.node.monetdb_interface.guard import is_valid_table_schema
from exareme2.node.monetdb_interface.guard import sql_injection_guard
from exareme2.node.monetdb_interface.monet_db_facade import db_execute_query
from exareme2.node_tasks_DTOs import TableSchema
from exareme2.node_tasks_DTOs import TableType


def get_table_names(context_id: str) -> List[str]:
    return common_actions.get_table_names(TableType.NORMAL, context_id)


@sql_injection_guard(table_name=str.isidentifier, table_schema=is_valid_table_schema)
def create_table(table_name: str, table_schema: TableSchema):
    columns_schema = convert_schema_to_sql_query_format(table_schema)
    db_execute_query(f"CREATE TABLE {table_name} ( {columns_schema} )")


@sql_injection_guard(table_name=str.isidentifier, table_values=None)
def insert_data_to_table(
    table_name: str, table_values: List[List[Union[str, int, float]]]
) -> None:
    # Ensure all rows have the same length
    row_length = len(table_values[0])
    column_length = len(table_values)
    if not all(len(row) == row_length for row in table_values):
        raise ValueError("All rows must have the same length")

    # Create the query parameters by flattening the list of rows
    parameters = [value for row in table_values for value in row]

    # Create the query with placeholders for each row value
    placeholders = ", ".join(
        ["(" + ", ".join(["%s"] * row_length) + ")"] * column_length
    )
    query = f"INSERT INTO {table_name} VALUES {placeholders}"

    # Execute the query with the parameters
    db_execute_query(query, parameters)

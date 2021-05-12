from typing import List, Union

from mipengine.common.node_tasks_DTOs import TableInfo
from mipengine.common.validate_identifier_names import validate_identifier_names
from mipengine.node.monetdb_interface import common_actions
from mipengine.node.monetdb_interface.common_actions import (
    convert_schema_to_sql_query_format,
)
from mipengine.node.monetdb_interface.monet_db_connection import MonetDB


def get_table_names(context_id: str) -> List[str]:
    return common_actions.get_table_names("normal", context_id)


@validate_identifier_names
def create_table(table_info: TableInfo):
    columns_schema = convert_schema_to_sql_query_format(table_info.schema)
    MonetDB().execute(f"CREATE TABLE {table_info.name} ( {columns_schema} )")


def convert_to_sql_string(value):
    if type(value) == str:
        return str(f"'{value}'")
    elif value:
        return str(value)
    else:
        return "null"


# TODO:Should validate the arguments, will be fixed with pydantic
def insert_data_to_table(table_name: str, rows: List[List[Union[str, int, float]]]):
    if all(len(row) != len(rows[0]) for row in rows):
        raise Exception("Row counts does not match")

    sql_values = []
    for row in rows:
        sql_row = ",".join([convert_to_sql_string(column) for column in row])
        sql_values.append(f"({sql_row})")

    MonetDB().execute(f"INSERT INTO {table_name} VALUES {','.join(sql_values)}")

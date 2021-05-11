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


# TODO:Should validate the arguments, will be fixed with pydantic
def insert_data_to_table(table_name: str, values: List[List[Union[str, int, float]]]):
    if all(len(value) != len(values[0]) for value in values):
        raise Exception("Row counts does not match")
    query_for_values = ",".join([str(tuple(value)) for value in values])
    query_for_values = str(query_for_values).replace(", None", ", null")
    MonetDB().execute(f"INSERT INTO {table_name} VALUES {query_for_values}")

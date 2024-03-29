from typing import Any
from typing import Dict
from typing import List
from typing import Union

import pymonetdb

from exareme2 import DType
from exareme2.worker import config as worker_config
from exareme2.worker.exareme2.monetdb.guard import is_list_of_identifiers
from exareme2.worker.exareme2.monetdb.guard import is_socket_address
from exareme2.worker.exareme2.monetdb.guard import is_valid_table_schema
from exareme2.worker.exareme2.monetdb.guard import sql_injection_guard
from exareme2.worker.exareme2.monetdb.monetdb_facade import db_execute_and_fetchall
from exareme2.worker.exareme2.monetdb.monetdb_facade import db_execute_query
from exareme2.worker_communication import ColumnData
from exareme2.worker_communication import ColumnDataBinary
from exareme2.worker_communication import ColumnDataFloat
from exareme2.worker_communication import ColumnDataInt
from exareme2.worker_communication import ColumnDataJSON
from exareme2.worker_communication import ColumnDataStr
from exareme2.worker_communication import ColumnInfo
from exareme2.worker_communication import IncompatibleSchemasMergeException
from exareme2.worker_communication import TableSchema
from exareme2.worker_communication import TablesNotFound
from exareme2.worker_communication import TableType


def create_table_name(
    table_type: TableType,
    node_id: str,
    context_id: str,
    command_id: str,
    result_id: str = "0",
) -> str:
    """
    Creates and returns in lower case a table name with the format
    <nodeId>_<contextId>_<tableType>_<commandId>_<command_subid>

    Underscores are not allowed in any parameter provided.
    """
    if not node_id.isalnum():
        raise ValueError(f"'node_id' is not alphanumeric. Value: '{node_id}'")
    if not context_id.isalnum():
        raise ValueError(f"'context_id' is not alphanumeric. Value: '{context_id}'")
    if not command_id.isalnum():
        raise ValueError(f"'command_id' is not alphanumeric. Value: '{command_id}'")
    if not result_id.isalnum():
        raise ValueError(f"'result_id' is not alphanumeric. Value: '{result_id}'")

    if table_type not in {TableType.NORMAL, TableType.VIEW, TableType.MERGE}:
        raise TypeError(f"Table type is not acceptable: {table_type} .")

    return f"{table_type}_{node_id}_{context_id}_{command_id}_{result_id}".lower()


@sql_injection_guard(table_type=None, context_id=str.isalnum)
def get_table_names(table_type: TableType, context_id: str) -> List[str]:
    """
    Retrieves a list of table names, which contain the context_id from the monetdb.

    Parameters
    ----------
    table_type : str
        The type of the table
    context_id : str
        The id of the experiment

    Returns
    ------
    List[str]
        A list of table names.
    """
    table_names = db_execute_and_fetchall(
        f"""
        SELECT name FROM tables
        WHERE
         type = {str(_convert_mip2monet_table_type(table_type))} AND
        name LIKE '%{context_id.lower()}%' AND
        system = false"""
    )

    return [table[0] for table in table_names]


def get_normal_table_names(context_id: str) -> List[str]:
    return get_table_names(TableType.NORMAL, context_id)


def get_merge_tables_names(context_id: str) -> List[str]:
    return get_table_names(TableType.MERGE, context_id)


def get_remote_table_names(context_id: str) -> List[str]:
    return get_table_names(TableType.REMOTE, context_id)


@sql_injection_guard(table_name=str.isidentifier)
def get_table_schema(table_name: str) -> TableSchema:
    """
    Retrieves a schema for a specific table name from the monetdb.

    Parameters
    ----------
    table_name : str
        The name of the table

    Returns
    ------
    TableSchema
        A schema which is TableSchema object.
    """
    schema = db_execute_and_fetchall(
        f"""
        SELECT columns.name, columns.type
        FROM columns
        RIGHT JOIN tables
        ON tables.id = columns.table_id
        WHERE
        tables.name = '{table_name}'
        """
    )
    if not schema:
        raise TablesNotFound([table_name])

    return TableSchema(
        columns=[
            ColumnInfo(
                name=name,
                dtype=DType.from_sql(sql_type=sql_type),
            )
            for name, sql_type in schema
        ]
    )


@sql_injection_guard(table_name=str.isidentifier)
def get_table_type(table_name: str) -> TableType:
    """
    Retrieves the type for a specific table name from the monetdb.

    Parameters
    ----------
    table_name : str
        The name of the table

    Returns
    ------
    TableType
        The type of the table.
    """

    monetdb_table_type_result = db_execute_and_fetchall(
        f"""
        SELECT type
        FROM
        tables
        WHERE
        tables.name = '{table_name}'
        """
    )
    if not monetdb_table_type_result:
        raise TablesNotFound([table_name])

    return _convert_monet2exareme2table_type(monetdb_table_type_result[0][0])


@sql_injection_guard(table_name=str.isidentifier, table_schema=is_valid_table_schema)
def create_table(table_name: str, table_schema: TableSchema):
    columns_schema = convert_schema_to_sql_query_format(table_schema)
    db_execute_query(f"CREATE TABLE {table_name} ( {columns_schema} )")


@sql_injection_guard(
    table_name=str.isidentifier,
    table_schema=is_valid_table_schema,
    merge_table_names=is_list_of_identifiers,
)
def create_merge_table(
    table_name: str,
    table_schema: TableSchema,
    merge_table_names: List[str],
):
    """
    The schema of the 1st table is used as the merge table schema.
    If there is an incompatibility or a table doesn't exist the db will throw an error.
    """
    columns_schema = convert_schema_to_sql_query_format(table_schema)
    merge_table_query = f"CREATE MERGE TABLE {table_name} ( {columns_schema} ); "
    for name in merge_table_names:
        merge_table_query += f"ALTER TABLE {table_name} ADD TABLE {name.lower()}; "

    try:
        db_execute_query(merge_table_query)
    except (
        pymonetdb.exceptions.ProgrammingError or pymonetdb.exceptions.OperationalError
    ) as exc:
        if str(exc).startswith("3F000"):
            raise IncompatibleSchemasMergeException(merge_table_names)
        if str(exc).startswith("42S02"):
            raise TablesNotFound(merge_table_names)
        else:
            raise exc


@sql_injection_guard(
    table_name=str.isidentifier,
    monetdb_socket_address=is_socket_address,
    schema=is_valid_table_schema,
    table_creator_username=str.isidentifier,
    public_username=str.isidentifier,
    public_password=str.isidentifier,
)
def create_remote_table(
    table_name: str,
    schema: TableSchema,
    monetdb_socket_address: str,
    table_creator_username: str,
    public_username: str,
    public_password: str,
):
    columns_schema = convert_schema_to_sql_query_format(schema)
    db_execute_query(
        f"""
        CREATE REMOTE TABLE {table_name}
        ( {columns_schema}) ON 'mapi:monetdb://{monetdb_socket_address}/db/{table_creator_username}/{table_name}'
        WITH USER '{public_username}' PASSWORD '{public_password}'
        """
    )


@sql_injection_guard(
    table_name=str.isidentifier,
    use_public_user=None,
)
def get_table_data(table_name: str, use_public_user: bool = True) -> List[ColumnData]:
    """
    Returns a list of columns data which will contain name, type and the data of the specific column.

    Parameters
    ----------
    table_name : str
        The name of the table
    use_public_user : bool
        Will the public or local user be used to access the data?

    Returns
    ------
    List[ColumnData]
        A list of column data
    """

    schema = get_table_schema(table_name)

    db_local_username = (
        worker_config.monetdb.local_username
    )  # The db local user, on whose namespace the tables are on.

    row_stored_data = db_execute_and_fetchall(
        f"SELECT * FROM {db_local_username}.{table_name}",
        use_public_user=use_public_user,
    )

    column_stored_data = list(zip(*row_stored_data))
    # In case there are no rows in the table, the `column_stored_data` will be an empty list.
    # The `column_stored_data` needs to have a value for each column, so we fill it with empty lists.
    if not column_stored_data:
        column_stored_data = [[] for _ in schema.columns]

    return _convert_column_stored_data_to_column_data_objects(
        column_stored_data, schema
    )


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


def _convert_column_stored_data_to_column_data_objects(
    column_stored_data: List[List[Any]], schema: TableSchema
):
    table_data = []
    for current_column, current_values in zip(schema.columns, column_stored_data):
        if current_column.dtype == DType.INT:
            table_data.append(
                ColumnDataInt(name=current_column.name, data=current_values)
            )
        elif current_column.dtype == DType.STR:
            table_data.append(
                ColumnDataStr(name=current_column.name, data=current_values)
            )
        elif current_column.dtype == DType.FLOAT:
            table_data.append(
                ColumnDataFloat(name=current_column.name, data=current_values)
            )
        elif current_column.dtype == DType.JSON:
            table_data.append(
                ColumnDataJSON(name=current_column.name, data=current_values)
            )
        elif current_column.dtype == DType.BINARY:
            table_data.append(
                ColumnDataBinary(name=current_column.name, data=current_values)
            )
        else:
            raise ValueError("Invalid column type")
    return table_data


def convert_schema_to_sql_query_format(schema: TableSchema) -> str:
    """
    Converts a table's schema to a sql query.

    Parameters
    ----------
    schema : TableSchema
        The schema of a table

    Returns
    ------
    str
        The schema in a sql query formatted string
    """

    return ", ".join(
        [f"{column.name} {column.dtype.to_sql()}" for column in schema.columns]
    )


def _convert_mip2monet_table_type(table_type: TableType) -> int:
    """
    Converts Exareme2's table types to MonetDB's table types
    """
    type_mapping = {
        TableType.NORMAL: 0,
        TableType.VIEW: 1,
        TableType.MERGE: 3,
        TableType.REMOTE: 5,
    }

    if table_type not in type_mapping.keys():
        raise ValueError(
            f"Type {table_type} cannot be converted to monetdb table type."
        )

    return type_mapping.get(table_type)


def _convert_monet2exareme2table_type(monet_table_type: int) -> TableType:
    """
    Converts MonetDB's table types to Exareme2's table types
    """
    type_mapping = {
        0: TableType.NORMAL,
        1: TableType.VIEW,
        3: TableType.MERGE,
        5: TableType.REMOTE,
    }

    if monet_table_type not in type_mapping.keys():
        raise ValueError(
            f"Type {monet_table_type} cannot be converted to Exareme2's table types."
        )

    return type_mapping.get(monet_table_type)


@sql_injection_guard(context_id=str.isalnum)
def get_tables_by_type(context_id: str) -> Dict[TableType, List[str]]:
    """
    Retrieve table names by type that contains a specific context_id from the DB.

    Parameters
    ----------
    context_id : str
        The id of the experiment
    """
    table_names_and_types = db_execute_and_fetchall(
        f"""
        SELECT name, type FROM tables
        WHERE name LIKE '%{context_id.lower()}%'
        AND system = false
        """
    )
    table_names_by_type = {}
    for table_type in TableType:
        table_names_by_type[table_type] = []

    for name, table_type in table_names_and_types:
        exareme2table_type = _convert_monet2exareme2table_type(table_type)
        table_names_by_type[exareme2table_type].append(name)
    return table_names_by_type

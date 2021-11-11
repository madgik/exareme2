from typing import List
from typing import Union

from mipengine import DType
from mipengine.node_exceptions import TablesNotFound
from mipengine.node_tasks_DTOs import ColumnInfo
from mipengine.node_tasks_DTOs import TableInfo
from mipengine.node_tasks_DTOs import TableSchema
from mipengine.node.monetdb_interface.monet_db_connection import MonetDB


# TODO We need to add the PRIVATE/OPEN table logic
from mipengine.node_tasks_DTOs import TableType


def create_table_name(
    table_type: TableType, command_id: str, context_id: str, node_id: str
) -> str:
    """
    Creates and returns in lower case a table name with the format <tableType>_<commandId>_<contextId>_<nodeId>
    """
    if table_type not in {TableType.NORMAL, TableType.VIEW, TableType.MERGE}:
        raise TypeError(f"Table type is not acceptable: {table_type} .")

    return f"{table_type}_{command_id}_{context_id}_{node_id}".lower()


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
    schema = MonetDB().execute_and_fetchall(
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

    monetdb_table_type_result = MonetDB().execute_and_fetchall(
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

    return _convert_monet2mip_table_type(monetdb_table_type_result[0][0])


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
    table_names = MonetDB().execute_and_fetchall(
        f"""
        SELECT name FROM tables
        WHERE
         type = {str(_convert_mip2monet_table_type(table_type))} AND
        name LIKE '%{context_id.lower()}%' AND
        system = false"""
    )

    return [table[0] for table in table_names]


def get_table_data(table_name: str) -> List[List[Union[float, int, str, None]]]:
    """
    Retrieves the data of a table with specific name from the monetdb.

    Parameters
    ----------
    table_name : str
        The name of the table

    Returns
    ------
    List[List[Union[str, int, float, bool]]
        The data of the table.
    """

    data = MonetDB().execute_and_fetchall(
        f"""
        SELECT {table_name}.*
        FROM {table_name}
        INNER JOIN tables ON tables.name = '{table_name}'
        WHERE tables.system=false
        """
    )

    return data


def get_initial_data_schemas() -> List[str]:
    """
    Retrieves all the different schemas that the initial datasets have.

    Returns
    ------
    List[str]
        The dataset schemas in the database.
    """

    schema_table_names = MonetDB().execute_and_fetchall(
        f"""
            SELECT name FROM tables
            WHERE name LIKE '%\\\\_data' ESCAPE '\\\\'
            AND system = false"""
    )

    # Flatten the list
    schema_table_names = [
        schema_table_name
        for schema_table in schema_table_names
        for schema_table_name in schema_table
    ]

    # The first part of the table is the dataset schema (pathology)
    # Table name convention = <schema_name>_data
    schema_names = [table_name.split("_")[0] for table_name in schema_table_names]

    return schema_names


def get_schema_datasets(schema_name) -> List[str]:
    """
    Retrieves the datasets with the specific schema.

    Returns
    ------
    List[str]
        The datasets of the schema.
    """

    datasets_rows = MonetDB().execute_and_fetchall(
        f"""
        SELECT DISTINCT(dataset) FROM {schema_name}_data
        """
    )

    # Flatten the list
    datasets = [
        dataset_name for dataset_row in datasets_rows for dataset_name in dataset_row
    ]

    return datasets


def drop_db_artifacts_by_context_id(context_id: str):
    """
    Drops all tables of any type and functions with name that contain a specific
    context_id from the DB.

    Parameters
    ----------
    context_id : str
        The id of the experiment
    """

    _drop_udfs_by_context_id(context_id)
    # Order of the table types matter not to have dependencies when dropping the tables
    for table_type in [
        TableType.MERGE,
        TableType.REMOTE,
        TableType.VIEW,
        TableType.NORMAL,
    ]:
        print("Dropping tabletype: " + str(table_type))
        _drop_table_by_type_and_context_id(table_type, context_id)


def _convert_monet2mip_table_type(monet_table_type: int) -> TableType:
    """
    Converts MonetDB's table types to MIP Engine's table types
    """
    type_mapping = {
        0: TableType.NORMAL,
        1: TableType.VIEW,
        3: TableType.MERGE,
        5: TableType.REMOTE,
    }

    if monet_table_type not in type_mapping.keys():
        raise ValueError(
            f"Type {monet_table_type} cannot be converted to MIP Engine's table types."
        )

    return type_mapping.get(monet_table_type)


def _convert_mip2monet_table_type(table_type: TableType) -> int:
    """
    Converts MIP Engine's table types to MonetDB's table types
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


def _drop_table_by_type_and_context_id(table_type: TableType, context_id: str):
    """
    Drops all tables of specific type with name that contain a specific context_id from the DB.

    Parameters
    ----------
    table_type : str
        The type of the table
    context_id : str
        The id of the experiment
    """
    table_names_and_types = MonetDB().execute_and_fetchall(
        f"""
        SELECT name, type FROM tables
        WHERE name LIKE '%{context_id.lower()}%'
        AND tables.type = {str(_convert_mip2monet_table_type(table_type))}
        AND system = false
        """
    )
    for name, table_type in table_names_and_types:
        if table_type == _convert_mip2monet_table_type(TableType.VIEW):
            MonetDB().execute(f"DROP VIEW {name}")
        else:
            MonetDB().execute(f"DROP TABLE {name}")


def _drop_udfs_by_context_id(context_id: str):
    """
    Drops all functions of specific context_id from the DB.

    Parameters
    ----------
    context_id : str
        The id of the experiment
    """
    function_names = MonetDB().execute_and_fetchall(
        f"""
        SELECT name FROM functions
        WHERE name LIKE '%{context_id.lower()}%'
        AND system = false
        """
    )
    for name in function_names:
        MonetDB().execute(f"DROP FUNCTION {name[0]}")

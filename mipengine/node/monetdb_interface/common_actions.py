import logging
from typing import Dict
from typing import List
from typing import Tuple

from mipengine import DType
from mipengine.node.monetdb_interface.monet_db_connection import MonetDB
from mipengine.node.monetdb_interface.monet_db_connection import monetdb
from mipengine.node_exceptions import TablesNotFound
from mipengine.node_tasks_DTOs import ColumnInfo
from mipengine.node_tasks_DTOs import CommonDataElement
from mipengine.node_tasks_DTOs import CommonDataElements
from mipengine.node_tasks_DTOs import TableSchema
from mipengine.node_tasks_DTOs import TableType
from mipengine.table_data_DTOs import ColumnData
from mipengine.table_data_DTOs import ColumnDataBinary
from mipengine.table_data_DTOs import ColumnDataFloat
from mipengine.table_data_DTOs import ColumnDataInt
from mipengine.table_data_DTOs import ColumnDataJSON
from mipengine.table_data_DTOs import ColumnDataStr


def create_table_name(
    table_type: TableType,
    node_id: str,
    context_id: str,
    command_id: str,
    command_subid: str = "0",
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
    if not command_subid.isalnum():
        raise ValueError(
            f"'command_subid' is not alphanumeric. Value: '{command_subid}'"
        )

    if table_type not in {TableType.NORMAL, TableType.VIEW, TableType.MERGE}:
        raise TypeError(f"Table type is not acceptable: {table_type} .")

    return f"{table_type}_{node_id}_{context_id}_{command_id}_{command_subid}".lower()


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
    schema = monetdb.execute_and_fetchall(
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

    monetdb_table_type_result = monetdb.execute_and_fetchall(
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
    table_names = monetdb.execute_and_fetchall(
        f"""
        SELECT name FROM tables
        WHERE
         type = {str(_convert_mip2monet_table_type(table_type))} AND
        name LIKE '%{context_id.lower()}%' AND
        system = false"""
    )

    return [table[0] for table in table_names]


def get_table_data(table_name: str) -> List[ColumnData]:
    """
    Returns a list of columns data which will contain name, type and the data of the specific column.

    Parameters
    ----------
    table_name : str
        The name of the table

    Returns
    ------
    List[ColumnData]
        A list of column data
    """
    schema = get_table_schema(table_name)

    data = monetdb.execute_and_fetchall(
        f"""
        SELECT {table_name}.*
        FROM {table_name}
        INNER JOIN tables ON tables.name = '{table_name}'
        WHERE tables.system=false
        """
    )

    # TableData contain columns
    # we need to switch the data given from the database from row-stored to column-stored
    data = list(zip(*data))

    columns_data = []
    for current_column, current_values in zip(schema.columns, data):
        if current_column.dtype == DType.INT:
            columns_data.append(
                ColumnDataInt(name=current_column.name, data=current_values)
            )
        elif current_column.dtype == DType.STR:
            columns_data.append(
                ColumnDataStr(name=current_column.name, data=current_values)
            )
        elif current_column.dtype == DType.FLOAT:
            columns_data.append(
                ColumnDataFloat(name=current_column.name, data=current_values)
            )
        elif current_column.dtype == DType.JSON:
            columns_data.append(
                ColumnDataJSON(name=current_column.name, data=current_values)
            )
        elif current_column.dtype == DType.BINARY:
            columns_data.append(
                ColumnDataBinary(name=current_column.name, data=current_values)
            )
        else:
            raise ValueError("Invalid column type")
    return columns_data


def get_data_models() -> List[str]:
    """
    Retrieves the enabled data_models from the database.

    Returns
    ------
    List[str]
        The data_models.
    """

    data_models_code_and_version = monetdb.execute_and_fetchall(
        f"""SELECT code, version
            FROM "mipdb_metadata"."data_models"
            WHERE status = 'ENABLED'
        """
    )
    data_models = [
        code + ":" + version for code, version in data_models_code_and_version
    ]
    return data_models


def get_dataset_code_per_dataset_label(data_model) -> Dict[str, str]:
    """
    Retrieves the enabled key-value pair of code and label, for a specific data_model.

    Returns
    ------
    Dict[str, str]
        The datasets.
    """
    data_model_code, data_model_version = data_model.split(":")

    datasets_rows = monetdb.execute_and_fetchall(
        f"""
        SELECT code, label
        FROM "mipdb_metadata"."datasets"
        WHERE data_model_id =
        (
            SELECT data_model_id
            FROM "mipdb_metadata"."data_models"
            WHERE code = '{data_model_code}'
            AND version = '{data_model_version}'
        )
        AND status = 'ENABLED'
        """
    )
    datasets = {code: label for code, label in datasets_rows}
    return datasets


def get_data_model_cdes(data_model) -> CommonDataElements:
    """
    Retrieves the cdes of the specific data_model.

    Returns
    ------
    CommonDataElements
        A CommonDataElements object
    """

    cdes_rows = monetdb.execute_and_fetchall(
        f"""
        SELECT code, metadata FROM "{data_model}"."variables_metadata"
        """
    )

    cdes = CommonDataElements(
        values={
            code: CommonDataElement.parse_raw(metadata) for code, metadata in cdes_rows
        }
    )

    return cdes


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
    table_names_and_types = monetdb.execute_and_fetchall(
        f"""
        SELECT name, type FROM tables
        WHERE name LIKE '%{context_id.lower()}%'
        AND tables.type = {str(_convert_mip2monet_table_type(table_type))}
        AND system = false
        """
    )
    for name, table_type in table_names_and_types:
        if table_type == _convert_mip2monet_table_type(TableType.VIEW):
            monetdb.execute(f"DROP VIEW {name}")
        else:
            monetdb.execute(f"DROP TABLE {name}")


def _drop_udfs_by_context_id(context_id: str):
    """
    Drops all functions of specific context_id from the DB.

    Parameters
    ----------
    context_id : str
        The id of the experiment
    """
    function_names = monetdb.execute_and_fetchall(
        f"""
        SELECT name FROM functions
        WHERE name LIKE '%{context_id.lower()}%'
        AND system = false
        """
    )
    for name in function_names:
        monetdb.execute(f"DROP FUNCTION {name[0]}")

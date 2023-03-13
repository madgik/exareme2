import json
from typing import Any
from typing import Dict
from typing import List

from mipengine import DType
from mipengine.exceptions import TablesNotFound
from mipengine.node.monetdb_interface.guard import is_datamodel
from mipengine.node.monetdb_interface.guard import sql_injection_guard
from mipengine.node.monetdb_interface.monet_db_facade import db_execute_and_fetchall
from mipengine.node.monetdb_interface.monet_db_facade import db_execute_query
from mipengine.node_tasks_DTOs import ColumnInfo
from mipengine.node_tasks_DTOs import CommonDataElement
from mipengine.node_tasks_DTOs import CommonDataElements
from mipengine.node_tasks_DTOs import DataModelAttributes
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

    return _convert_monet2mip_table_type(monetdb_table_type_result[0][0])


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


@sql_injection_guard(table_name=str.isidentifier)
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
    # TODO: blocked by https://team-1617704806227.atlassian.net/browse/MIP-133 .
    # Retrieving the data should be a simple select.
    # row_stored_data = db_execute_and_fetchall(f"SELECT * FROM {table_name}")

    row_stored_data = db_execute_and_fetchall(
        f"""
        SELECT {table_name}.*
        FROM {table_name}
        INNER JOIN tables ON tables.name = '{table_name}'
        WHERE tables.system=false
        """
    )

    column_stored_data = list(zip(*row_stored_data))
    # In case there are no rows in the table, the `column_stored_data` will be an empty list.
    # The `column_stored_data` needs to have a value for each column, so we fill it with empty lists.
    if not column_stored_data:
        column_stored_data = [[] for _ in schema.columns]

    return _convert_column_stored_data_to_column_data_objects(
        column_stored_data, schema
    )


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


def get_data_models() -> List[str]:
    """
    Retrieves the enabled data_models from the database.

    Returns
    ------
    List[str]
        The data_models.
    """

    data_models_code_and_version = db_execute_and_fetchall(
        f"""SELECT code, version
            FROM "mipdb_metadata"."data_models"
            WHERE status = 'ENABLED'
        """
    )
    data_models = [
        code + ":" + version for code, version in data_models_code_and_version
    ]
    return data_models


@sql_injection_guard(data_model=is_datamodel)
def get_dataset_code_per_dataset_label(data_model: str) -> Dict[str, str]:
    """
    Retrieves the enabled key-value pair of code and label, for a specific data_model.

    Returns
    ------
    Dict[str, str]
        The datasets.
    """
    data_model_code, data_model_version = data_model.split(":")

    datasets_rows = db_execute_and_fetchall(
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


@sql_injection_guard(data_model=is_datamodel)
def get_data_model_cdes(data_model: str) -> CommonDataElements:
    """
    Retrieves the cdes of the specific data_model.

    Returns
    ------
    CommonDataElements
        A CommonDataElements object
    """
    data_model_code, data_model_version = data_model.split(":")

    cdes_rows = db_execute_and_fetchall(
        f"""
        SELECT code, metadata FROM "{data_model_code}:{data_model_version}"."variables_metadata"
        """
    )

    cdes = CommonDataElements(
        values={
            code: CommonDataElement.parse_raw(metadata) for code, metadata in cdes_rows
        }
    )

    return cdes


@sql_injection_guard(data_model=is_datamodel)
def get_data_model_attributes(data_model: str) -> DataModelAttributes:
    """
    Retrieves the attributes, for a specific data_model.

    Returns
    ------
    DataModelAttributes
    """
    data_model_code, data_model_version = data_model.split(":")

    attributes = db_execute_and_fetchall(
        f"""
        SELECT properties
        FROM "mipdb_metadata"."data_models"
        WHERE code = '{data_model_code}'
        AND version = '{data_model_version}'
        """
    )

    attributes = json.loads(attributes[0][0])
    return DataModelAttributes(
        tags=attributes["tags"], properties=attributes["properties"]
    )


def drop_db_artifacts_by_context_id(context_id: str):
    """
    Drops all tables of any type and functions with name that contain a specific
    context_id from the DB.

    Parameters
    ----------
    context_id : str
        The id of the experiment
    """

    function_names = _get_function_names_query_by_context_id(context_id)

    udfs_deletion_query = _get_drop_udfs_query(function_names)
    table_names_by_type = _get_tables_by_type(context_id)
    tables_deletion_query = _get_drop_tables_query(table_names_by_type)
    db_execute_query(udfs_deletion_query + tables_deletion_query)


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


@sql_injection_guard(context_id=str.isalnum)
def _get_tables_by_type(context_id: str) -> Dict[TableType, List[str]]:
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
        mip_table_type = _convert_monet2mip_table_type(table_type)
        table_names_by_type[mip_table_type].append(name)
    return table_names_by_type


@sql_injection_guard(context_id=str.isalnum)
def _get_function_names_query_by_context_id(context_id: str) -> List[str]:
    """
    Retrieve all functions of specific context_id from the DB.

    Parameters
    ----------
    context_id : str
        The id of the experiment
    """
    result = db_execute_and_fetchall(
        f"""
        SELECT name FROM functions
        WHERE name LIKE '%{context_id.lower()}%'
        AND system = false
        """
    )
    return [attributes[0] for attributes in result]


def _get_drop_udfs_query(function_names: List[str]):
    return "".join([f"DROP FUNCTION {name};" for name in function_names])


def _get_drop_tables_query(table_names_by_type: Dict[TableType, List[str]]):
    # Order of the table types matter not to have dependencies when dropping the tables
    table_type_drop_order = (
        TableType.MERGE,
        TableType.REMOTE,
        TableType.VIEW,
        TableType.NORMAL,
    )
    table_deletion_query = "".join(
        [
            _get_drop_table_query(table_name, table_type)
            for table_type in table_type_drop_order
            for table_name in table_names_by_type[table_type]
        ]
    )
    return table_deletion_query


def _get_drop_table_query(table_name, table_type) -> str:
    return (
        f"DROP VIEW {table_name};"
        if table_type == TableType.VIEW
        else f"DROP TABLE {table_name};"
    )

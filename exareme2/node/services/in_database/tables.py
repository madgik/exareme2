from typing import List

from exareme2.node import config as node_config
from exareme2.node.logger import initialise_logger
from exareme2.node.monetdb import tables
from exareme2.node.monetdb.tables import create_table_name
from exareme2.node_communication import TableData
from exareme2.node_communication import TableInfo
from exareme2.node_communication import TableSchema
from exareme2.node_communication import TableType


@initialise_logger
def get_tables(request_id: str, context_id: str) -> List[str]:
    """
    Parameters
    ----------
    request_id : str
        The identifier for the logging
    context_id : str
    The id of the experiment

    Returns
    ------
    List[str]
        A list of table names
    """
    return tables.get_normal_table_names(context_id)


@initialise_logger
def get_remote_tables(request_id: str, context_id: str) -> List[str]:
    """
    Parameters
    ----------
    request_id : str
        The identifier for the logging
    context_id : str
    The id of the experiment

    Returns
    ------
    List[str]
        A list of remote table names
    """
    return tables.get_remote_table_names(context_id)


@initialise_logger
def get_merge_tables(request_id: str, context_id: str) -> List[str]:
    """
    Parameters
    ----------
    request_id : str
        The identifier for the logging
    context_id : str
        The id of the experiment

    Returns
    ------
    List[str]
        A list of merge table names
    """
    return tables.get_merge_tables_names(context_id)


@initialise_logger
def create_table(
    request_id: str, context_id: str, command_id: str, table_schema: TableSchema
) -> TableInfo:
    """
    Parameters
    ----------
    request_id : str
        The identifier for the logging.
    context_id : str
        The id of the experiment.
    command_id : str
        The id of the command that the table.
    table_schema : TableSchema
        A TableSchema object.
    """
    table_name = create_table_name(
        TableType.NORMAL,
        node_config.identifier,
        context_id,
        command_id,
    )
    tables.create_table(table_name, table_schema)

    return TableInfo(
        name=table_name,
        schema_=table_schema,
        type_=TableType.NORMAL,
    )


@initialise_logger
def create_remote_table(
    request_id: str,
    table_name: str,
    table_schema: TableSchema,
    monetdb_socket_address: str,
):
    """
    Parameters
    ----------
    request_id : str
        The identifier for the logging.
    table_name : str
        The name of the table.
    table_schema : TableSchema
        A TableSchema object.
    monetdb_socket_address : str
        The monetdb_socket_address of the monetdb that we want to create the remote table from.
    """
    local_username = node_config.monetdb.local_username
    public_username = node_config.monetdb.public_username
    public_password = node_config.monetdb.public_password
    tables.create_remote_table(
        table_name=table_name,
        schema=table_schema,
        monetdb_socket_address=monetdb_socket_address,
        table_creator_username=local_username,
        public_username=public_username,
        public_password=public_password,
    )


@initialise_logger
def create_merge_table(
    request_id: str, context_id: str, command_id: str, table_infos: List[TableInfo]
) -> TableInfo:
    """
    Parameters
    ----------
    request_id : str
        The identifier for the logging.
    context_id : str
        The id of the experiment.
    command_id : str
        The id of the command that the merge table.
    table_infos: List[str(TableInfo)]
        A list of TableInfo of the tables to be merged.
    """
    merge_table_name = create_table_name(
        TableType.MERGE,
        node_config.identifier,
        context_id,
        command_id,
    )

    tables.create_merge_table(
        table_name=merge_table_name,
        table_schema=table_infos[0].schema_,
        merge_table_names=[table_info.name for table_info in table_infos],
    )

    return TableInfo(
        name=merge_table_name,
        schema_=table_infos[0].schema_,
        type_=TableType.MERGE,
    )


@initialise_logger
def get_table_data(request_id: str, table_name: str) -> TableData:
    """
    Parameters
    ----------
    request_id : str
        The identifier for the logging
    table_name : str
        The name of the table
    """
    # If the public user is used, its ensured that the table won't hold private data.
    # Tables are published to the public DB user when they are meant for sending to other nodes.
    # The "protect_local_data" config allows for turning this logic off in testing scenarios.
    use_public_user = True if node_config.privacy.protect_local_data else False

    columns = tables.get_table_data(table_name, use_public_user)

    return TableData(name=table_name, columns=columns)

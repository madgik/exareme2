from typing import Dict
from typing import List

from exareme2.worker.exareme2.monetdb.guard import sql_injection_guard
from exareme2.worker.exareme2.monetdb.monetdb_facade import db_execute_and_fetchall
from exareme2.worker.exareme2.monetdb.monetdb_facade import db_execute_query
from exareme2.worker.exareme2.tables.tables_db import get_tables_by_type
from exareme2.worker_communication import TableType


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
    table_names_by_type = get_tables_by_type(context_id)
    tables_deletion_query = _get_drop_tables_query(table_names_by_type)
    db_execute_query(udfs_deletion_query + tables_deletion_query)


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

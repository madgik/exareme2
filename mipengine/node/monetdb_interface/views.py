from typing import List

from pymonetdb import Connection
from pymonetdb.sql.cursors import Cursor

from mipengine.common.validate_identifier_names import validate_identifier_names
from mipengine.node.monetdb_interface import common_actions
from mipengine.node.monetdb_interface.connection_pool import execute_with_occ


def get_views_names(cursor: Cursor, context_id: str) -> List[str]:
    return common_actions.get_tables_names(cursor, "view", context_id)


@validate_identifier_names
def create_view(connection:Connection, cursor: Cursor, view_name: str, pathology: str, datasets: List[str], columns: List[str]):
    # TODO: Add filters argument
    dataset_names = ','.join(f"'{dataset}'" for dataset in datasets)
    columns = ', '.join(columns)
    execute_with_occ(connection, cursor,
        f"CREATE VIEW {view_name} AS SELECT {columns} FROM {pathology}_data WHERE dataset IN ({dataset_names})")

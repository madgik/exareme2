from typing import List

from mipengine.common.validate_identifier_names import validate_identifier_names
from mipengine.node.monetdb_interface import common_action
from mipengine.node.monetdb_interface.common_action import connection
from mipengine.node.monetdb_interface.common_action import cursor

DATA_TABLE_PRIMARY_KEY = 'row_id'


def get_views_names(context_id: str) -> List[str]:
    return common_action.get_tables_names("view", context_id)


@validate_identifier_names
def create_view(view_name: str, pathology: str, datasets: List[str], columns: List[str]):
    # TODO: Add filters argument
    dataset_names = ','.join(f"'{dataset}'" for dataset in datasets)
    columns = ', '.join(columns)
    cursor.execute(
        f"""CREATE VIEW {view_name} 
        AS SELECT {DATA_TABLE_PRIMARY_KEY}, {columns} 
        FROM {pathology}_data 
        WHERE dataset IN ({dataset_names})"""
    )
    connection.commit()

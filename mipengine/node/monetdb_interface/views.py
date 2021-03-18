from typing import List

from mipengine.common.validate_identifier_names import validate_identifier_names
from mipengine.node.monetdb_interface import common_actions
from mipengine.node.monetdb_interface.monet_db_connection import execute_with_occ

DATA_TABLE_PRIMARY_KEY = 'row_id'


def get_views_names(context_id: str) -> List[str]:
    return common_actions.get_tables_names("view", context_id)


@validate_identifier_names
def create_view(view_name: str, pathology: str, datasets: List[str], columns: List[str]):
    # TODO: Add filters argument
    dataset_names = ','.join(f"'{dataset}'" for dataset in datasets)
    columns = ', '.join(columns)

    execute_with_occ(
        f"""CREATE VIEW {view_name} 
        AS SELECT {DATA_TABLE_PRIMARY_KEY}, {columns} 
        FROM {pathology}_data 
        WHERE dataset IN ({dataset_names})"""
    )

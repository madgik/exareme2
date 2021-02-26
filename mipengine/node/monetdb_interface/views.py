from typing import List
from typing import Union

from mipengine.node.monetdb_interface import common
from mipengine.node.monetdb_interface.common import connection
from mipengine.node.monetdb_interface.common import cursor
from mipengine.utils.validate_identifier_names import validate_identifier_names


def get_views_names(context_id: str) -> List[str]:
    return common.get_tables_names("view", context_id)


@validate_identifier_names
def create_view(view_name: str, pathology: str, datasets: List[str], columns: List[str]):
    # TODO: Add filters argument
    dataset_names = ','.join(f"'{dataset}'" for dataset in datasets)

    cursor.execute(
        f"CREATE VIEW {view_name} AS SELECT {', '.join(columns)} FROM {pathology}_data WHERE dataset IN ({dataset_names})")
    connection.commit()

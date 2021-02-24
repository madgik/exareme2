from typing import List
from typing import Union

from mipengine.node.monetdb_interface import common
from mipengine.node.monetdb_interface.common import connection
from mipengine.node.monetdb_interface.common import cursor
from mipengine.node.tasks.data_classes import TableSchema
from mipengine.utils.validate_identifier_names import validate_identifier_names


def get_views_names(context_id: str) -> List[str]:
    return common.get_tables_names("view", context_id)


def get_view_schema(view_name: str) -> TableSchema:
    return common.get_table_schema('view', view_name)


def get_view_data(context_id: str) -> List[List[Union[str, int, float, bool]]]:
    return common.get_table_data("view", context_id)


@validate_identifier_names
def create_view(view_name: str, pathology: str, datasets: List[str], columns: List[str], filters_json: str):
    cursor.execute(
        f"CREATE VIEW {view_name} AS SELECT {', '.join(columns)} FROM {pathology}_data WHERE dataset IN ({str(datasets)[1:-1]})")
    connection.commit()

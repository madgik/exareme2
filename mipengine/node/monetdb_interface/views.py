from typing import List
from typing import Union

from mipengine.node.monetdb_interface import common
from mipengine.node.monetdb_interface.common import connection
from mipengine.node.monetdb_interface.common import cursor
from mipengine.node.tasks.data_classes import TableSchema
from mipengine.utils.verify_identifier_names import sql_injections_defender


def get_views_names(context_id: str) -> List[str]:
    return common.get_tables_names("view", context_id)


def get_view_schema(view_name: str) -> TableSchema:
    return common.get_table_schema('view', view_name)


def get_view_data(context_id: str) -> List[List[Union[str, int, float, bool]]]:
    return common.get_table_data("view", context_id)


@sql_injections_defender
def create_view(view_name: str, columns: List[str], datasets: List[str]):
    cursor.execute(
        f"CREATE VIEW {view_name} AS SELECT {', '.join(columns)} FROM data WHERE dataset IN ({str(datasets)[1:-1]})")
    connection.commit()

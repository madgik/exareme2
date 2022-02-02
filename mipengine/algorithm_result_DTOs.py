from typing import List
from typing import Union

from mipengine.node_tasks_DTOs import ImmutableBaseModel

from mipengine.table_data_DTOs import ColumnDataFloat
from mipengine.table_data_DTOs import ColumnDataInt
from mipengine.table_data_DTOs import ColumnDataStr


class TabularDataResult(ImmutableBaseModel):
    title: str
    columns: List[Union[ColumnDataInt, ColumnDataStr, ColumnDataFloat]]

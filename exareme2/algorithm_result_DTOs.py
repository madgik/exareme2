from typing import List
from typing import Union

from exareme2.node_tasks_DTOs import ImmutableBaseModel
from exareme2.table_data_DTOs import ColumnDataFloat
from exareme2.table_data_DTOs import ColumnDataInt
from exareme2.table_data_DTOs import ColumnDataStr


class TabularDataResult(ImmutableBaseModel):
    title: str
    columns: List[Union[ColumnDataInt, ColumnDataStr, ColumnDataFloat]]

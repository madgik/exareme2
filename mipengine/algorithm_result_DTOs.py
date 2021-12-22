from typing import List
from typing import Union

from pydantic import BaseModel

from mipengine.table_data_DTOs import ColumnDataFloat
from mipengine.table_data_DTOs import ColumnDataInt
from mipengine.table_data_DTOs import ColumnDataStr


class TabularDataResult(BaseModel):
    title: str
    columns: List[Union[ColumnDataInt, ColumnDataStr, ColumnDataFloat]]

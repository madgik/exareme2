from typing import List, Union
from pydantic import BaseModel


class TabularDataColumn(BaseModel):
    name: str
    type: str


class TabularDataResult(BaseModel):
    title: str
    columns: List[TabularDataColumn]
    data: List[List[Union[float, str]]]

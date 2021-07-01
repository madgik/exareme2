from typing import List, Union
from pydantic import BaseModel


class TabularDataResult(BaseModel):
    title: str
    columns: List[str]
    data: List[List[Union[float, str]]]

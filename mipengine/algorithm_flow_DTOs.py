from typing import List, Union, Any

from mipengine.node_tasks_DTOs import ImmutableBaseModel
from mipengine.node_tasks_DTOs import UDFArgumentKind


class TabularDataColumn(ImmutableBaseModel):
    name: str
    type: str


class TabularDataResult(ImmutableBaseModel):
    title: str
    columns: List[TabularDataColumn]
    data: List[List[Union[float, str]]]


class Literal(ImmutableBaseModel):
    value: Any
    kind = UDFArgumentKind.LITERAL

    class Config:
        allow_mutation = False

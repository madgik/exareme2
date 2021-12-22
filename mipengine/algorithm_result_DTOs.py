from typing import List, Union
from pydantic import BaseModel
from pydantic import validator

from mipengine import DType


class _ColumnData(BaseModel):
    name: str
    type: DType

    @validator("type")
    def validate_type(cls, tp):
        if cls.__name__ == "ColumnData":
            raise TypeError(
                "ColumnData should not be instantiated. "
                "Use ColumnDataInt, ColumnDataStr or ColumnDataFloat instead."
            )
        column_type = cls.__fields__["type"].default
        if tp != column_type:
            raise ValueError(
                f"Objects of type {cls.__name__} have a fixed type {column_type}, "
                f"you cannot use {tp} in the constructor."
            )
        return tp


class ColumnDataInt(_ColumnData):
    data: List[Union[None, int]]
    type = DType.INT


class ColumnDataStr(_ColumnData):
    data: List[Union[None, str]]
    type = DType.STR


class ColumnDataFloat(_ColumnData):
    data: List[Union[None, float]]
    type = DType.FLOAT


class TabularDataResult(BaseModel):
    title: str
    columns: List[Union[ColumnDataInt, ColumnDataStr, ColumnDataFloat]]

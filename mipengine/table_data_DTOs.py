from abc import ABC
from typing import List
from typing import Union

from pydantic import BaseModel
from pydantic import validator
from typing import Any

from mipengine import DType


class ImmutableBaseModel(BaseModel, ABC):
    class Config:
        allow_mutation = False


class _ColumnData(ImmutableBaseModel):
    name: str
    data: List[Any]
    type: DType

    @validator("type")
    def validate_type(cls, tp):
        if cls.__name__ == "_ColumnData":
            raise TypeError(
                "ColumnData should not be instantiated. "
                "Use ColumnDataInt, ColumnDataStr, ColumnDataFloat, ColumnDataJSON  or ColumnDataBinary instead."
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


class ColumnDataJSON(_ColumnData):
    data: List[Union[None, str]]
    type = DType.JSON


class ColumnDataBinary(_ColumnData):
    data: List[Union[None, int]]
    type = DType.BINARY

from warnings import warn
import enum
from typing import (
    Any,
    Dict,
    List,
    Tuple,
    Union,
    Optional,
)

from pydantic import (
    BaseModel,
    validator,
    Field,
)

# ~~~~~~~~~~~~~~~~~~~~ Enums ~~~~~~~~~~~~~~~~~~~~ #


class DBDataType(enum.Enum):
    INT = enum.auto()
    FLOAT = enum.auto()
    TEXT = enum.auto()


class UDFArgumentKind(enum.Enum):
    TABLE = enum.auto()
    LITERAL = enum.auto()


# ~~~~~~~~~~~~~~~~~~ Validators ~~~~~~~~~~~~~~~~~ #


def validate_name(name):
    if not name.isidentifier():
        raise ValueError(f"Expected valid identifier, got {name}")
    if not name.islower():
        warn(f"Names must be lowercase, got {name}")
        return name.lower()
    return name


# ~~~~~~~~~~~~~~~~~~~ DTOs ~~~~~~~~~~~~~~~~~~~~~~ #


class ColumnInfo(BaseModel):
    name: str
    dtype: DBDataType

    _validate_name = validator("name", allow_reuse=True)(validate_name)


class TableSchema(BaseModel):
    columns: List[ColumnInfo]


class TableInfo(BaseModel):
    name: str
    schema_: TableSchema

    _validate_name = validator("name", allow_reuse=True)(validate_name)


class TableView(BaseModel):
    datasets: List[str]
    columns: List[str]
    filter: Optional[dict]

    _validate_names = validator(
        "datasets",
        "columns",
        each_item=True,
        allow_reuse=True,
    )(validate_name)


class TableData(BaseModel):
    schema_: TableSchema
    data: List[Tuple[Union[str, int, float, bool]]]


class UDFArgument(BaseModel):
    kind: UDFArgumentKind
    value: Any


class CategoricalFieldEnum(BaseModel):
    level: str
    label: str

    _validate_name = validator("level", allow_reuse=True)(validate_name)


class DataFieldMetadata(BaseModel):
    name: str
    label: str
    dtype: DBDataType
    is_categorical: bool
    enumerations: Optional[List[CategoricalFieldEnum]]
    min_: Optional[float]
    max_: Optional[float]

    _validate_name = validator("name", allow_reuse=True)(validate_name)


class AlgorithmInputData(BaseModel):
    pathology: str
    datasets: List[str]
    filter: Optional[dict]
    var_groups: Dict[str, List[str]]


class AlgorithmRequest(BaseModel):
    inputdata: AlgorithmInputData
    parameters: Optional[Dict[str, Any]]

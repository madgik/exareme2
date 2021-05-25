from warnings import warn
import enum
from typing import (
    Any,
    Dict,
    List,
    Tuple,
    Union,
    Optional,
    Set
)

from pydantic import (
    BaseModel,
    validator,
    Field,
    StrictStr,
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
    data: List[Tuple[Union[StrictStr, int, float, bool]]]


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
    pathology: StrictStr
    datasets: List[StrictStr]
    filter: Optional[dict]
    var_groups: Dict[StrictStr, List[StrictStr]]


class AlgorithmRequest(BaseModel):
    inputdata: AlgorithmInputData
    parameters: Optional[Dict[StrictStr, Any]]


class MetadataEnumeration(BaseModel):
    code: StrictStr
    label: StrictStr


class MetadataVariable(BaseModel):
    code: StrictStr
    label: StrictStr
    dtype: DBDataType
    isCategorical: bool
    enumerations: Optional[List[MetadataEnumeration]] = None
    min: Optional[float] = None
    max: Optional[float] = None


class MetadataGroup(BaseModel):
    """
    MetadataGroup is used to map the pathology metadata .json to an object.
    """
    code: StrictStr
    label: StrictStr
    variables: Optional[List[MetadataVariable]] = Field(default_factory=list)
    groups: Optional[List["MetadataGroup"]] = Field(default_factory=list)

class CommonDataElement(BaseModel):
    label: StrictStr
    dtype: DBDataType
    categorical: bool
    enumerations: Optional[Set] = None
    min: Optional[float] = None
    max: Optional[float] = None


class CommonDataElements(BaseModel):
    pathologies: Dict[StrictStr, Dict[StrictStr, CommonDataElement]]


# ~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~ #
import pytest
from pydantic import ValidationError

@pytest.fixture
def column_info_name():
    return "lala"

@pytest.fixture
def column_info_data_type():
    return DBDataType.INT

# asserts correct parameters in test_column_info
def test_column_info(column_info_name, column_info_data_type):
    expected_name = 'lala'
    expected_dt = column_info_data_type
    assert expected_name == column_info_name
    assert expected_dt == column_info_data_type

@pytest.fixture
def column_info_name_error():
    return '8493hfsk'

@pytest.fixture
def column_info_dtype_error():
    return 3.4j

# raises error for wrong params in ColumnInfo
# @pytest.mark.xfail
def test_column_info_error(column_info_name_error, column_info_dtype_error):
    with pytest.raises(ValidationError) as exception:
        ColumnInfo(name = column_info_name_error, dtype = column_info_dtype_error)


@pytest.fixture
def table_schema_cols():
    cols = [ColumnInfo(name = 'Layla', dtype = DBDataType.FLOAT),
            ColumnInfo(name = 34j, dtype = DBDataType.FLOAT)]
    return cols


# asserts correct parameters in test_column_info
def test_table_schema():
    cols = [ColumnInfo(name = 'layla', dtype = DBDataType.FLOAT),
            ColumnInfo(name = 'dsafdsa', dtype = DBDataType.FLOAT)]
    assert isinstance(cols, List)
    assert isinstance(cols[0], ColumnInfo)
    assert isinstance(cols[1], ColumnInfo)


# validation check for TableSchema with error
def test_table_schema_error():
    with pytest.raises(ValidationError):
        list = [ColumnInfo(name = 'Layla', dtype = DBDataType.FLOAT),
                ColumnInfo(name = 34j, dtype = DBDataType.FLOAT)]

@pytest.fixture
def table_info_data_name():
    return 'this is a string'

@pytest.fixture
def table_info_data_schema():
    return TableSchema(columns=[ColumnInfo(name = 'layla', dtype = DBDataType.FLOAT),
                                ColumnInfo(name = 'sheila', dtype = DBDataType.FLOAT)])


# validation check for table_info
def test_table_info(table_info_data_name, table_info_data_schema):
    assert table_info_data_name == 'this is a string'
    assert isinstance(table_info_data_schema, TableSchema)



@pytest.fixture
def table_info_error_str():
    return DBDataType.FLOAT


@pytest.fixture
def table_info_data_schema_error():
    return TableSchema(columns=[ColumnInfo(name = DBDataType.FLOAT, dtype = DBDataType.FLOAT),
            ColumnInfo(name = 'Sheila', dtype = DBDataType.FLOAT)])

# validation check for table_info
def test_table_info_error(table_info_data_name):
    with pytest.raises(ValidationError) as e:
        TableInfo(name=table_info_error_str, schema_=table_info_data_schema_error)

def test_table_view_error():
    with pytest.raises(ValidationError) as e:
        TableView(datasets=[34, 'bar', 'baz'],
                  columns=[],
                  filter=[])

def test_table_data_error():
    with pytest.raises(ValidationError) as e:
        TableData(schema_='foo', data=34)

def test_table_data():
    with pytest.raises(ValidationError) as e:
        TableData(schema_= 'this is not a TableSchema object', data='and this is not a list of tuples')

def test_udf_argument():
    with pytest.raises(ValidationError) as e:
        UDFArgument(kind='Not a UDFArgumentKind', value='this can be anything')

def test_categorical_field_enum():
    with pytest.raises(ValidationError) as e:
        CategoricalFieldEnum(level=34, label=42)

def test_data_field_metadata():
    # Should warn for indentifier
    with pytest.raises(ValidationError) as e:
        DataFieldMetadata(name=34,
                          label=34,
                          dtype=1,
                          is_categorical=False)
    print(e.value.json())

def test_algorithm_input_data():
    with pytest.raises(ValidationError) as e:
        AlgorithmInputData(pathology='dementia',
                           datasets='ppmi',
                           filter={},
                           var_groups= 'wrong')

def test_algorithn_request():
    with pytest.raises(ValidationError) as e:
        AlgorithmRequest(inputdata='wrong')


def test_metadata_enumeration():
    with pytest.raises(ValidationError) as e:
        MetadataEnumeration(code='a code', label=34)


def test_metadata_variable():
    with pytest.raises(ValidationError) as e:
        MetadataVariable(code=34,
                         label=34,
                         dtype=3,
                         isCategorical=True)


def test_metadata_group():
    with pytest.raises(ValidationError) as e:
        MetadataGroup(code='lala',
                      label=32)


def test_common_data_element():
    with pytest.raises(ValidationError) as e:
        CommonDataElement(label=32,
                          dtype=3,
                          categorical=False)


def test_common_data_elements():
    with pytest.raises(ValidationError) as e:
        CommonDataElements(pathologies='oh, well')

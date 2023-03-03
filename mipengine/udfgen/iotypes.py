from abc import ABC
from abc import abstractproperty
from typing import TypeVar

from mipengine import DType as dt
from mipengine.udfgen.helpers import iotype_to_sql_schema
from mipengine.udfgen.helpers import recursive_repr

LN = "\n"
ROWID = "row_id"
DEFERRED = "deferred"


# ~~~~~~~~~~~~~~~~~~~~~~~ IO Types ~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class ParametrizedType:
    @property
    def known_typeparams(self):
        typeparams = self.__dict__
        return {
            name: typeparam
            for name, typeparam in typeparams.items()
            if not isinstance(typeparam, TypeVar)
        }

    @property
    def unknown_typeparams(self):
        typeparams = self.__dict__
        return {
            name: typeparam
            for name, typeparam in typeparams.items()
            if isinstance(typeparam, TypeVar)
        }

    @property
    def is_generic(self):
        return len(self.unknown_typeparams) > 0


class IOType(ABC):
    __repr__ = recursive_repr

    def __eq__(self, other) -> bool:
        return self.__dict__ == other.__dict__


class InputType(IOType):
    pass


class OutputType(IOType):
    @abstractproperty
    def schema(self):
        raise NotImplementedError


class LoopbackOutputType(OutputType):
    pass


class UDFLoggerType(InputType):
    pass


def udf_logger():
    return UDFLoggerType()


class PlaceholderType(InputType):
    def __init__(self, name):
        self.name = "$" + name


def placeholder(name):
    """
    UDF input type factory for inserting an assignment to arbitrary an
    placeholder in UDF definition.

    Examples
    --------
    Using this function in the udf decorator, as show below

    >>> @udf(x=placeholder("some_name"))
    ... def f(x):
    ...    pass

    will insert the line

        x = $some_name

    in the SQL UDF definition.
    """
    return PlaceholderType(name)


# special type for passing MIN_ROW_COUNT in UDF. Only Node knows the actual
# value so here it's exported as a placeholder and replaced by Node.
MIN_ROW_COUNT = placeholder("min_row_count")


class TableType(ABC):
    @abstractproperty
    def schema(self):
        raise NotImplementedError

    def column_names(self, prefix=""):
        prefix += "_" if prefix else ""
        return [prefix + name for name, _ in self.schema]

    def get_return_type_template(self):
        return f"TABLE({iotype_to_sql_schema(self)})"


class TensorType(TableType, ParametrizedType, InputType, OutputType):
    def __init__(self, dtype, ndims):
        self.dtype = dt.from_py(dtype) if isinstance(dtype, type) else dtype
        self.ndims = ndims

    @property
    def schema(self):
        dimcolumns = [(f"dim{i}", dt.INT) for i in range(self.ndims)]
        valcolumn = [("val", self.dtype)]
        return dimcolumns + valcolumn

    def get_build_template(self) -> str:
        columns_tmpl = "{{name: _columns[name_w_prefix] for name, name_w_prefix in zip({colnames}, {colnames_w_prefix})}}"
        return f"{{varname}} = udfio.from_tensor_table({columns_tmpl})"

    def get_main_return_stmt_template(self) -> str:
        return "return udfio.as_tensor_table(numpy.array({return_name}))"


def tensor(dtype, ndims):
    return TensorType(dtype, ndims)


class MergeTensorType(TableType, ParametrizedType, InputType, OutputType):
    def __init__(self, dtype, ndims):
        self.dtype = dt.from_py(dtype) if isinstance(dtype, type) else dtype
        self.ndims = ndims

    @property
    def schema(self):
        dimcolumns = [(f"dim{i}", dt.INT) for i in range(self.ndims)]
        valcolumn = [("val", self.dtype)]
        return dimcolumns + valcolumn  # type: ignore

    def get_build_template(self) -> str:
        colums_tmpl = "{{name: _columns[name_w_prefix] for name, name_w_prefix in zip({colnames}, {colnames_w_prefix})}}"
        return f"{{varname}} = udfio.merge_tensor_to_list({colums_tmpl})"


def merge_tensor(dtype, ndims):
    return MergeTensorType(dtype, ndims)


class RelationType(TableType, ParametrizedType, InputType, OutputType):
    def __init__(self, schema):
        error_msg = (
            "Expected schema of type TypeVar, DEFFERED or List[Tuple]. "
            f"Got {schema}."
        )
        if isinstance(schema, TypeVar):
            self._schema = schema
        elif schema == DEFERRED:
            self._schema = DEFERRED
        elif isinstance(schema, list):
            # Subscripted generics cannot be used with class and instance
            # checks. This means that we have to check if the list has the
            # correct structure. The only acceptable structure is an empty list
            # or a list of pairs.
            if schema == []:
                pass
            elif isinstance(schema[0], tuple) and len(schema[0]) == 2:
                pass
            else:
                raise TypeError(error_msg)
            self._schema = [
                (name, self._convert_dtype(dtype)) for name, dtype in schema
            ]
        else:
            raise TypeError(error_msg)

    @staticmethod
    def _convert_dtype(dtype):
        if isinstance(dtype, type):
            return dt.from_py(dtype)
        if isinstance(dtype, str):
            return dt(dtype)
        if isinstance(dtype, dt):
            return dtype
        raise TypeError(f"Expected dtype of type type, str or DType. Got {dtype}.")

    @property
    def schema(self):
        return self._schema

    def get_build_template(self) -> str:
        columns_tmpl = "{{name: _columns[name_w_prefix] for name, name_w_prefix in zip({colnames}, {colnames_w_prefix})}}"
        return f"{{varname}} = udfio.from_relational_table({columns_tmpl}, '{ROWID}')"

    def get_main_return_stmt_template(self) -> str:
        return f"return udfio.as_relational_table({{return_name}}, '{ROWID}')"


def relation(schema=None):
    schema = schema or TypeVar("S")
    return RelationType(schema)


class DictType(TableType, ABC):
    _data_column_name: str
    _data_column_type: dt

    @property
    def data_column_name(self):
        return self._data_column_name

    @property
    def data_column_type(self):
        return self._data_column_type

    @property
    def schema(self):
        return [(self.data_column_name, self.data_column_type)]


class TransferType(DictType, InputType, LoopbackOutputType):
    _data_column_name = "transfer"
    _data_column_type = dt.JSON

    def get_build_template(self) -> str:
        colname = self.data_column_name
        return LN.join(
            [
                f'__transfer_str = _conn.execute("SELECT {colname} from {{table_name}};")["{colname}"][0]',
                "{varname} = json.loads(__transfer_str)",
            ]
        )

    def get_main_return_stmt_template(self) -> str:
        return "return json.dumps({return_name})"

    def get_secondary_return_stmt_template(self, tablename_placeholder) -> str:
        return (
            '_conn.execute(f"INSERT INTO '
            + tablename_placeholder
            + " VALUES ('{{json.dumps({return_name})}}');\")"
        )


def transfer():
    return TransferType()


class MergeTransferType(DictType, InputType):
    _data_column_name = "transfer"
    _data_column_type = dt.JSON

    def get_build_template(self) -> str:
        colname = self.data_column_name
        return LN.join(
            [
                f'__transfer_strs = _conn.execute("SELECT {colname} from {{table_name}};")["{colname}"]',
                "{varname} = [json.loads(str) for str in __transfer_strs]",
            ]
        )


def merge_transfer():
    return MergeTransferType()


class TransferTypeBase(ABC):
    pass


TransferTypeBase.register(TransferType)
TransferTypeBase.register(MergeTransferType)


class StateType(DictType, InputType, LoopbackOutputType):
    _data_column_name = "state"
    _data_column_type = dt.BINARY

    def get_build_template(self):
        colname = self.data_column_name
        return LN.join(
            [
                f'__state_str = _conn.execute("SELECT {colname} from {{table_name}};")["{colname}"][0]',
                "{varname} = pickle.loads(__state_str)",
            ]
        )

    def get_main_return_stmt_template(self) -> str:
        return "return pickle.dumps({return_name})"

    def get_secondary_return_stmt_template(self, tablename_placeholder) -> str:
        return (
            '_conn.execute(f"INSERT INTO '
            + tablename_placeholder
            + " VALUES ('{{pickle.dumps({return_name}).hex()}}');\")"
        )


def state():
    return StateType()


class LiteralType(InputType):
    pass


def literal():
    return LiteralType()


# ~~~~~~~~~~~~~~~~~~~~~~~ UDF Arguments ~~~~~~~~~~~~~~~~~~~~~~ #


class UDFArgument:
    __repr__ = recursive_repr
    type: InputType


class UDFLoggerArg(UDFArgument):
    type = UDFLoggerType()
    udf_name: str

    def __init__(self, udf_name="", request_id=""):
        self.udf_name = udf_name
        self.request_id = request_id


class PlaceholderArg(UDFArgument):
    def __init__(self, type):
        self.type = type

    @property
    def name(self):
        return self.type.name


class TableArg(UDFArgument, ABC):
    type: TableType

    def __init__(self, table_name):
        self.table_name = table_name

    def column_names(self, prefix=""):
        return self.type.column_names(prefix)


class TensorArg(TableArg):
    def __init__(self, table_name, dtype, ndims):
        self.type: TensorType = tensor(dtype, ndims)
        super().__init__(table_name)

    @property
    def ndims(self):
        return self.type.ndims

    @property
    def dtype(self):
        return self.type.dtype

    def __eq__(self, other):
        if self.table_name != other.table_name:
            return False
        if self.dtype != other.dtype:
            return False
        if self.ndims != other.ndims:
            return False
        return True


class RelationArg(TableArg):
    def __init__(self, table_name, schema):
        self.type = relation(schema)
        super().__init__(table_name)

    @property
    def schema(self):
        return self.type.schema

    def __eq__(self, other):
        if self.table_name != other.table_name:
            return False
        if self.schema != other.schema:
            return False
        return True


class DictArg(TableArg, ABC):
    type: DictType

    def __init__(self, table_name: str):
        super().__init__(table_name)

    @property
    def schema(self):
        return self.type.schema

    def __eq__(self, other):
        if self.table_name != other.table_name:
            return False
        if self.schema != other.schema:
            return False
        return True


class StateArg(DictArg):
    type = state()

    def __init__(self, table_name: str):
        super().__init__(table_name)


class TransferArg(DictArg):
    type = transfer()

    def __init__(self, table_name: str):
        super().__init__(table_name)


class LiteralArg(UDFArgument):
    def __init__(self, value):
        self._value = value
        self.type: LiteralType = literal()

    @property
    def value(self):
        return self._value

    def __eq__(self, other):
        return self.value == other.value

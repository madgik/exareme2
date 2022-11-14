from abc import ABC
from abc import abstractproperty
from typing import TypeVar

from mipengine import DType as dt
from mipengine.udfgen.helpers import iotype_to_sql_schema
from mipengine.udfgen.helpers import recursive_repr

LN = "\n"
MAIN_TABLE_PLACEHOLDER = "main_output_table_name"
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

    def get_main_return_stmt_template(self, _) -> str:
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

    def get_main_return_stmt_template(self, _) -> str:
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

    def get_main_return_stmt_template(self, _) -> str:
        return "return json.dumps({return_name})"

    def get_secondary_return_stmt_template(self, tablename_placeholder, _) -> str:
        return (
            '_conn.execute(f"INSERT INTO $'
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


class SecureTransferType(DictType, InputType, LoopbackOutputType):
    _data_column_name = "secure_transfer"
    _data_column_type = dt.JSON
    _sum_op: bool
    _min_op: bool
    _max_op: bool

    def __init__(self, sum_op=False, min_op=False, max_op=False):
        self._sum_op = sum_op
        self._min_op = min_op
        self._max_op = max_op

    @property
    def sum_op(self):
        return self._sum_op

    @property
    def min_op(self):
        return self._min_op

    @property
    def max_op(self):
        return self._max_op

    def get_build_template(self):
        colname = self.data_column_name
        return LN.join(
            [
                f'__transfer_strs = _conn.execute("SELECT {colname} from {{table_name}};")["{colname}"]',
                "__transfers = [json.loads(str) for str in __transfer_strs]",
                "{varname} = udfio.secure_transfers_to_merged_dict(__transfers)",
            ]
        )

    def get_main_return_stmt_template(self, smpc_used) -> str:
        return self._get_secure_transfer_main_return_stmt_template(smpc_used)

    def _get_secure_transfer_main_return_stmt_template(self, smpc_used):
        if smpc_used:
            return_stmts = [
                "template, sum_op, min_op, max_op = udfio.split_secure_transfer_dict({return_name})"
            ]
            (
                _,
                sum_op_tmpl,
                min_op_tmpl,
                max_op_tmpl,
            ) = _get_smpc_table_template_names(MAIN_TABLE_PLACEHOLDER)
            return_stmts.extend(
                self._get_secure_transfer_op_return_stmt_template(
                    self.sum_op, sum_op_tmpl, "sum_op"
                )
            )
            return_stmts.extend(
                self._get_secure_transfer_op_return_stmt_template(
                    self.min_op, min_op_tmpl, "min_op"
                )
            )
            return_stmts.extend(
                self._get_secure_transfer_op_return_stmt_template(
                    self.max_op, max_op_tmpl, "max_op"
                )
            )
            return_stmts.append("return json.dumps(template)")
            return LN.join(return_stmts)
        else:
            # Treated as a TransferType
            return "return json.dumps({return_name})"

    def get_secondary_return_stmt_template(
        self, tablename_placeholder, smpc_used
    ) -> str:
        return self._get_secure_transfer_sec_return_stmt_template(
            tablename_placeholder, smpc_used
        )

    def _get_secure_transfer_sec_return_stmt_template(
        self, tablename_placeholder: str, smpc_used
    ):
        if smpc_used:
            return_stmts = [
                "template, sum_op, min_op, max_op = udfio.split_secure_transfer_dict({return_name})"
            ]
            (
                template_tmpl,
                sum_op_tmpl,
                min_op_tmpl,
                max_op_tmpl,
            ) = _get_smpc_table_template_names(tablename_placeholder)
            return_stmts.append(
                '_conn.execute(f"INSERT INTO $'
                + template_tmpl
                + " VALUES ('{{json.dumps(template)}}');\")"
            )
            return_stmts.extend(
                self._get_secure_transfer_op_return_stmt_template(
                    self.sum_op, sum_op_tmpl, "sum_op"
                )
            )
            return_stmts.extend(
                self._get_secure_transfer_op_return_stmt_template(
                    self.min_op, min_op_tmpl, "min_op"
                )
            )
            return_stmts.extend(
                self._get_secure_transfer_op_return_stmt_template(
                    self.max_op, max_op_tmpl, "max_op"
                )
            )
            return LN.join(return_stmts)
        else:
            # Treated as a TransferType
            return (
                '_conn.execute(f"INSERT INTO $'
                + tablename_placeholder
                + " VALUES ('{{json.dumps({return_name})}}');\")"
            )

    @staticmethod
    def _get_secure_transfer_op_return_stmt_template(
        op_enabled, table_name_tmpl, op_name
    ):
        if not op_enabled:
            return []
        return [
            '_conn.execute(f"INSERT INTO $'
            + table_name_tmpl
            + f" VALUES ('{{{{json.dumps({op_name})}}}}');\")"
        ]

    def get_smpc_build_template(self):
        def get_smpc_op_template(enabled, operation_name):
            stmts = []
            if enabled:
                stmts.append(
                    f'__{operation_name}_values_str = _conn.execute("SELECT secure_transfer from {{{operation_name}_values_table_name}};")["secure_transfer"][0]'
                )
                stmts.append(
                    f"__{operation_name}_values = json.loads(__{operation_name}_values_str)"
                )
            else:
                stmts.append(f"__{operation_name}_values = None")
            return stmts

        stmts = []
        stmts.append(
            '__template_str = _conn.execute("SELECT secure_transfer from {template_table_name};")["secure_transfer"][0]'
        )
        stmts.append("__template = json.loads(__template_str)")
        stmts.extend(get_smpc_op_template(self.sum_op, "sum_op"))
        stmts.extend(get_smpc_op_template(self.min_op, "min_op"))
        stmts.extend(get_smpc_op_template(self.max_op, "max_op"))
        stmts.append(
            "{varname} = udfio.construct_secure_transfer_dict(__template,__sum_op_values,__min_op_values,__max_op_values)"
        )
        return LN.join(stmts)


def _get_smpc_table_template_names(prefix: str):
    """
    This is used when a secure transfer is returned with smpc enabled.
    The secure_transfer is one output_type but needs to be broken into
    multiple tables, hence more than one main table names are needed.
    """
    return (
        prefix,
        prefix + "_sum_op",
        prefix + "_min_op",
        prefix + "_max_op",
    )


def secure_transfer(sum_op=False, min_op=False, max_op=False):
    if not sum_op and not min_op and not max_op:
        raise ValueError(
            "In a secure_transfer at least one operation should be enabled."
        )
    return SecureTransferType(sum_op, min_op, max_op)


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

    def get_main_return_stmt_template(self, _) -> str:
        return "return pickle.dumps({return_name})"

    def get_secondary_return_stmt_template(self, tablename_placeholder, _) -> str:
        return (
            '_conn.execute(f"INSERT INTO $'
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

    def __init__(self, udf_name):
        self.udf_name = udf_name


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


class SecureTransferArg(DictArg):
    type = SecureTransferType()

    def __init__(self, table_name: str):
        super().__init__(table_name)


class SMPCSecureTransferArg(UDFArgument):
    type: SecureTransferType
    template_table_name: str
    sum_op_values_table_name: str
    min_op_values_table_name: str
    max_op_values_table_name: str

    def __init__(
        self,
        template_table_name: str,
        sum_op_values_table_name: str,
        min_op_values_table_name: str,
        max_op_values_table_name: str,
    ):
        sum_op = False
        min_op = False
        max_op = False
        if sum_op_values_table_name:
            sum_op = True
        if min_op_values_table_name:
            min_op = True
        if max_op_values_table_name:
            max_op = True
        self.type = SecureTransferType(sum_op, min_op, max_op)
        self.template_table_name = template_table_name
        self.sum_op_values_table_name = sum_op_values_table_name
        self.min_op_values_table_name = min_op_values_table_name
        self.max_op_values_table_name = max_op_values_table_name


class LiteralArg(UDFArgument):
    def __init__(self, value):
        self._value = value
        self.type: LiteralType = literal()

    @property
    def value(self):
        return self._value

    def __eq__(self, other):
        return self.value == other.value

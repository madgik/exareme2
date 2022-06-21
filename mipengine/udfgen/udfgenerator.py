"""
Module for generating SQL code for defining and calling MonetDB/Python UDFs

This module provides functions/classes for:
    - Registering python functions as MonetDB UDFs
    - Translating python functions into MonetDB UDF definitions
    - Generating SQL code for using MonetDB UDFs
    - Generating SQL code for certain tensor operations


1. Registering a python function as a MonetDB UDF
-------------------------------------------------

A python function can be registered as a MonetDB UDF using the udf decorator.
A number of conditions must be met for a successful registration:
    - All parameters need to be annotated using the factory functions found in
    the table below
    - The return type must also be declared using the same factory functions
    - The function needs to have only one return statement
    - Only a name can be returned, no expressions allowed
If any of the above is violated a UDFBadDefinition exception is raised.

The factory functions return objects inheriting from IOType which is an
abstract class representing all UDF input/output types. These types are
parametrized with various type parameters. The type parameters can be concrete
values or instances of TypeVar. If all type parameters are concrete values the
IOType is said to be *specific*, else the IOType is said to be *generic*.

UDF Example (specific)
~~~~~~~~~~~~~~~~~~~~~~

>>> @udf(x=tensor(dtype=int, ndims=1), return_type=scalar(dtype=int))
... def sum_vector(x):
...     result = sum(x)
...     return result

Let's look at the decorator

@udf(x=tensor(dtype=int, ndims=1), return_type=scalar(dtype=int))
     ^                             ^
     |                             |
     The function takes an         The function returns a
     argument x of type            "scalar of datatype int"
     "one dimensional tensor
     of datatype int"

Notice that the type of x is more than just a datatype. It is a more complex
notion, here a tensor of dimensions 1, holding elements of datatype int. We
need this level of detail for the translation process.

UDF Example (generic)
~~~~~~~~~~~~~~~~~~~~~

>>> T = TypeVar('T')
>>> N = TypeVar('N')
>>> @udf(
...      x=tensor(dtype=T, ndims=N),
...      y=tensor(dtype=T, ndims=N),
...      return_type=tensor(dtype=T, ndims=N)
... )
... def tensor_diff(x, y):
...     result = x - y
...     return result

First notice that we need to define some generic type parameters. These are
used instead of concrete values to signal that a type parameter value is not
known at compile time but will become known at runtime.

At runtime, concrete arguments will be passed to the UDF translator. The type
variables T and N will then be mapped to the concrete values found in the
arguments. In the return_type annotation we use the same type variables (T and
N) as in the input types. This means that their values will be inferred at
runtime from the corresponding values passed in the arguments. For example if
at runtime x and y are tensors of type float and dimensions 2 then the return
type will be a tensor of type float and dimensions 2 as well.

Tensors and Relations explained
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are two main kinds of UDF input/output type, tensors and relations.  Both
are just tables in the DB. However, they are understood differently by the UDF
translator/generator.

Relations are formally sets of tuples. Hence the tuple order is immaterial.  An
example of a relation is a collection of medical records.

|------------+--------+-----+-----|
| patient_id | gender | age | ... |
|------------+--------+-----+-----|
|          1 | F      |  60 |     |
|          2 | M      |  75 |     |
|          3 | F      |  42 |     |
|------------+--------+-----+-----|

Tensors are formally lists of lists of lists... The nesting level depends of
the tensor dimensions. They differ from relations in two crucial ways:
    1. Order matters
    2. Any number of dimensions is allowed, not just two

Since tensors will be also written as tables in the DB a special representation
is required to capture the structure described above.

Example:
The two dimensional tensor (matrix)
    [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]
becomes
    |------+------+-----|
    | dim0 | dim1 | val |
    |------+------+-----|
    |    0 |    0 |   1 |
    |    0 |    1 |   2 |
    |    0 |    2 |   3 |
    |    1 |    0 |   4 |
    |    1 |    1 |   5 |
    |    1 |    2 |   6 |
    |    2 |    0 |   7 |
    |    2 |    1 |   8 |
    |    2 |    2 |   9 |
    |------+------+-----|

The first two columns are the indices for the two dimensions and the third
column is the value found at the position specified by the respective indices.

State and Transfer explained
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
State and Transfer are special input/output types. They are materialized as a
simple dict within the udf, where the user is free to insert any variable.
States are then saved locally on the same node, to be used in later udfs,
whereas Transfers are always transferred to the opposite node.

======================= ==================== =====================
                        State                Transfer
======================= ==================== =====================
Type in udf             Dict                 Dict
Type in DB              BINARY               CLOB
Encoding using          Pickle               Json
Shareable               no                   yes
Input Type              yes                  yes
Output Type             yes                  yes
======================= ==================== =====================


Local/Global steps explained
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For State and Transfer there is also the option of multiple return values.
This is used for the local/global step logic where a local step (udf) keeps a State
locally and sends a Transfer to the global node. Respectively, a global step (udf)
keeps a State globally and sends a Transfer object to all the local nodes.

Local UDF step Example
~~~~~~~~~~~~~~~~~~~~~~
>>> @udf(x=state(), y=transfer(), return_type=[state(), transfer()])
... def local_step(x, y):
...     state["x"] = x["key"]
...     state["y"] = y["key"]
...     transfer["sum"] = x["key"] + y["key"]
...     return state, transfer

Global UDF step Example
~~~~~~~~~~~~~~~~~~~~~~
>>> @udf(x=state(), y=merge_transfer(), return_type=[state(), transfer()])
... def global_step(x, y):
...     state["x"] = x["key"]
...     sum = 0
...     for transfer in y:
...         sum += transfer["key"]
...     state["y"] = sum
...     transfer["sum"] = x["key"] + y["key"]
...     return state, transfer

NOTE: Even though it's not mandatory, it's faster to have state as the first input/output.


secure_transfer explained
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
There is also the option of using secure_transfer instead of transfer
if you want the algorithm to support integration with the SMPC cluster.
The local/global udfs steps using secure_transfer would look like:
Local UDF step Example
~~~~~~~~~~~~~~~~~~~~~~
>>> @udf(x=state(), y=transfer(), return_type=[state(), secure_transfer()])
... def local_step(x, y):
...     state["x"] = x["key"]
...     state["y"] = y["key"]
...     transfer["sum"] = {"data": x["key"] + y["key"], "operation": "sum", "type": "float"}
...     return state, transfer

Global UDF step Example
~~~~~~~~~~~~~~~~~~~~~~
>>> @udf(x=state(), y=secure_transfer(), return_type=[state(), transfer()])
... def global_step(x, y):
...     state["x"] = x["key"]
...     sum = y["sum"]      # The values from all the local nodes are already aggregated
...     state["y"] = sum
...     transfer["sum"] = x["key"] + y["key"]
...     return state, transfer

So, the secure_transfer dict sent should be of the format:
>>> {
...    "data": DATA,
...    "type": TYPE,
...    "operation": OPERATION
... }

The data could be an int/float or a list containing other lists or float/int.

The operation enumerations are:
    - "sum" (Floats not supported when SMPC is enabled)
    - "min" (Floats not supported when SMPC is enabled)
    - "max" (Floats not supported when SMPC is enabled)
    - "union" (Not yet supported)


2. Translating and calling UDFs
-------------------------------

Once a UDF is registered we can generate a pair of strings using the function
generate_udf_queries. The first string contains SQL code with
definition of the UDF. The second string is a small script with a CREATE TABLE
query followed by a INSERT/SELECT query. The CREATE TABLE query creates the
table which will hold the UDF's result and the INSERT/SELECT query calls the
UDF and inserts the result into the result table.


3. Generating SQL code for tensor operations
--------------------------------------------

The representation of tensor described in the first section allows us to write
certain tensor operation directly as SQL queries, without using the MonetDB
Python UDF functionality.

These operations are:
    - Elementwise arithmetic operations (+, -, *, /)
    - Dot products
    - Matrix transpotitions

The module exports two enums: TensorUnaryOp and TensorBinaryOp whose
enumerations represent the above operations.

The code generation process is the same as before with the difference that the
first string returned by generate_udf_queries is empty since there
is no UDF definition. The second string holds the queries for executing the
operation and inserting the result into a new table.





All the module's exposed objects are found in the table below.

======================= ========================================================
Exposed object          Description
======================= ========================================================
udf                     Decorator for registering python funcs as UDFs
tensor                  Tensor type factory
relation                Relation type factory
merge_tensor            Merge tensor type factory
scalar                  Scalar type factory
literal                 Literal type factory
state                   State type factory
transfer                Transfer type factory
merge_transfer          Merge transfer type factory
generate_udf_queries    Generates a pair of strings holding the UDF definition
                        (when needed) and the query for calling the UDF
TensorUnaryOp           Enum with tensor unary operations
TensorBinaryOp          Enum with tensor binary operations
make_unique_func_name   Helper for creating unique function names
======================= ========================================================
"""
import ast
import base64
import hashlib
import inspect
import re
from abc import ABC
from abc import abstractmethod
from copy import deepcopy
from enum import Enum
from numbers import Number
from string import Template
from textwrap import dedent
from textwrap import indent
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union

import astor
import numpy

from mipengine import DType as dt
from mipengine.node_tasks_DTOs import TableInfo
from mipengine.node_tasks_DTOs import TableType as DBTableType
from mipengine.udfgen.udfgen_DTOs import SMPCTablesInfo
from mipengine.udfgen.udfgen_DTOs import SMPCUDFGenResult
from mipengine.udfgen.udfgen_DTOs import TableUDFGenResult
from mipengine.udfgen.udfgen_DTOs import UDFGenExecutionQueries
from mipengine.udfgen.udfgen_DTOs import UDFGenResult

__all__ = [
    "udf",
    "udf_logger",
    "tensor",
    "relation",
    "merge_tensor",
    "scalar",
    "literal",
    "transfer",
    "merge_transfer",
    "state",
    "secure_transfer",
    "generate_udf_queries",
    "TensorUnaryOp",
    "TensorBinaryOp",
    "make_unique_func_name",
]

# TODO Do not select with star, select columns explicitly to avoid surprises
# with node_id etc.

# TODO need a class TypeParameter so that ParametrizedType knows which params
# to classify into known/unknown. That way, extra params can be added in the
# future without breaking the known/unknown mechanism.

# ~~~~~~~~~~~~~~~~~~~ SQL Tokens and Templates ~~~~~~~~~~~~~~~ #

CREATE_OR_REPLACE_FUNCTION = "CREATE OR REPLACE FUNCTION"
RETURNS = "RETURNS"
LANGUAGE_PYTHON = "LANGUAGE PYTHON"
BEGIN = "{"
END = "}"

SELECT = "SELECT"
FROM = "FROM"
WHERE = "WHERE"
GROUP_BY = "GROUP BY"
ORDER_BY = "ORDER BY"
DROP_TABLE_IF_EXISTS = "DROP TABLE IF EXISTS"
CREATE_TABLE = "CREATE TABLE"
AND = "AND"

SEP = ","
LN = "\n"
SEPLN = SEP + LN
ANDLN = " " + AND + LN
SPC4 = " " * 4
SCOLON = ";"
ROWID = "row_id"


def get_smpc_build_template(secure_transfer_type):
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
    stmts.extend(get_smpc_op_template(secure_transfer_type.sum_op, "sum_op"))
    stmts.extend(get_smpc_op_template(secure_transfer_type.min_op, "min_op"))
    stmts.extend(get_smpc_op_template(secure_transfer_type.max_op, "max_op"))
    stmts.append(
        "{varname} = udfio.construct_secure_transfer_dict(__template,__sum_op_values,__min_op_values,__max_op_values)"
    )
    return LN.join(stmts)


# TODO refactor these, polymorphism?
# Currently DictTypes are loaded with loopback queries only.
def get_table_build_template(input_type: "TableType"):
    if not isinstance(input_type, InputType):
        raise TypeError(
            f"Build template only for InputTypes. Type provided: {input_type}"
        )
    COLUMNS_COMPREHENSION_TMPL = "{{n: _columns[n] for n in {colnames}}}"
    if isinstance(input_type, RelationType):
        return f"{{varname}} = pd.DataFrame({COLUMNS_COMPREHENSION_TMPL})"
    if isinstance(input_type, TensorType):
        return f"{{varname}} = udfio.from_tensor_table({COLUMNS_COMPREHENSION_TMPL})"
    if isinstance(input_type, MergeTensorType):
        return f"{{varname}} = udfio.merge_tensor_to_list({COLUMNS_COMPREHENSION_TMPL})"
    if isinstance(input_type, MergeTransferType):
        colname = input_type.data_column_name
        loopback_query = f'__transfer_strs = _conn.execute("SELECT {colname} from {{table_name}};")["{colname}"]'
        dict_parse = "{varname} = [json.loads(str) for str in __transfer_strs]"
        return LN.join([loopback_query, dict_parse])
    if isinstance(input_type, TransferType):
        colname = input_type.data_column_name
        loopback_query = f'__transfer_str = _conn.execute("SELECT {colname} from {{table_name}};")["{colname}"][0]'
        dict_parse = "{varname} = json.loads(__transfer_str)"
        return LN.join([loopback_query, dict_parse])
    if isinstance(input_type, SecureTransferType):
        colname = input_type.data_column_name
        loopback_query = f'__transfer_strs = _conn.execute("SELECT {colname} from {{table_name}};")["{colname}"]'
        dict_parse = "__transfers = [json.loads(str) for str in __transfer_strs]"
        transfers_data_aggregation = (
            "{varname} = udfio.secure_transfers_to_merged_dict(__transfers)"
        )
        return LN.join([loopback_query, dict_parse, transfers_data_aggregation])
    if isinstance(input_type, StateType):
        colname = input_type.data_column_name
        loopback_query = f'__state_str = _conn.execute("SELECT {colname} from {{table_name}};")["{colname}"][0]'
        dict_parse = "{varname} = pickle.loads(__state_str)"
        return LN.join([str(loopback_query), dict_parse])
    raise NotImplementedError(
        f"Type {input_type} doesn't have a build statement template."
    )


def _get_secure_transfer_op_return_stmt_template(op_enabled, table_name_tmpl, op_name):
    if not op_enabled:
        return []
    return [
        '_conn.execute(f"INSERT INTO $'
        + table_name_tmpl
        + f" VALUES ('$node_id', '{{{{json.dumps({op_name})}}}}');\")"
    ]


def _get_secure_transfer_main_return_stmt_template(output_type, smpc_used):
    if smpc_used:
        return_stmts = [
            "template, sum_op, min_op, max_op = udfio.split_secure_transfer_dict({return_name})"
        ]
        (
            _,
            sum_op_tmpl,
            min_op_tmpl,
            max_op_tmpl,
        ) = _get_smpc_table_template_names(_get_main_table_template_name())
        return_stmts.extend(
            _get_secure_transfer_op_return_stmt_template(
                output_type.sum_op, sum_op_tmpl, "sum_op"
            )
        )
        return_stmts.extend(
            _get_secure_transfer_op_return_stmt_template(
                output_type.min_op, min_op_tmpl, "min_op"
            )
        )
        return_stmts.extend(
            _get_secure_transfer_op_return_stmt_template(
                output_type.max_op, max_op_tmpl, "max_op"
            )
        )
        return_stmts.append("return json.dumps(template)")
        return LN.join(return_stmts)
    else:
        # Treated as a TransferType
        return "return json.dumps({return_name})"


def get_main_return_stmt_template(output_type: "OutputType", smpc_used: bool):
    if not isinstance(output_type, OutputType):
        raise TypeError(
            f"Return statement template only for OutputTypes. Type provided: {output_type}"
        )
    if isinstance(output_type, RelationType):
        return "return udfio.as_relational_table(numpy.array({return_name}))"
    if isinstance(output_type, TensorType):
        return "return udfio.as_tensor_table(numpy.array({return_name}))"
    if isinstance(output_type, ScalarType):
        return "return {return_name}"
    if isinstance(output_type, StateType):
        return "return pickle.dumps({return_name})"
    if isinstance(output_type, TransferType):
        return "return json.dumps({return_name})"
    if isinstance(output_type, SecureTransferType):
        return _get_secure_transfer_main_return_stmt_template(output_type, smpc_used)
    raise NotImplementedError(
        f"Type {output_type} doesn't have a return statement template."
    )


def _get_secure_transfer_sec_return_stmt_template(
    output_type, tablename_placeholder: str, smpc_used
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
            + " VALUES ('$node_id', '{{json.dumps(template)}}');\")"
        )
        return_stmts.extend(
            _get_secure_transfer_op_return_stmt_template(
                output_type.sum_op, sum_op_tmpl, "sum_op"
            )
        )
        return_stmts.extend(
            _get_secure_transfer_op_return_stmt_template(
                output_type.min_op, min_op_tmpl, "min_op"
            )
        )
        return_stmts.extend(
            _get_secure_transfer_op_return_stmt_template(
                output_type.max_op, max_op_tmpl, "max_op"
            )
        )
        return LN.join(return_stmts)
    else:
        # Treated as a TransferType
        return (
            '_conn.execute(f"INSERT INTO $'
            + tablename_placeholder
            + " VALUES ('$node_id', '{{json.dumps({return_name})}}');\")"
        )


def get_secondary_return_stmt_template(
    output_type: "OutputType", tablename_placeholder: str, smpc_used: bool
):
    if not isinstance(output_type, LoopbackOutputType):
        raise TypeError(
            f"Secondary return statement template only for LoopbackOutputTypes. Type provided: {output_type}"
        )
    if isinstance(output_type, TransferType):
        return (
            '_conn.execute(f"INSERT INTO $'
            + tablename_placeholder
            + " VALUES ('$node_id', '{{json.dumps({return_name})}}');\")"
        )
    if isinstance(output_type, StateType):
        return (
            '_conn.execute(f"INSERT INTO $'
            + tablename_placeholder
            + " VALUES ('$node_id', '{{pickle.dumps({return_name}).hex()}}');\")"
        )
    if isinstance(output_type, SecureTransferType):
        return _get_secure_transfer_sec_return_stmt_template(
            output_type, tablename_placeholder, smpc_used
        )
    raise NotImplementedError(
        f"Type {output_type} doesn't have a loopback return statement template."
    )


def get_return_type_template(output_type):
    if not isinstance(output_type, OutputType):
        raise TypeError(
            f"Return type template only for OutputTypes. Type provided: {output_type}"
        )
    if isinstance(output_type, TableType):
        return f"TABLE({iotype_to_sql_schema(output_type)})"
    if isinstance(output_type, ScalarType):
        return output_type.dtype.to_sql()
    raise NotImplementedError(
        f"Type {output_type} doesn't have a return type template."
    )


def iotype_to_sql_schema(iotype, name_prefix=""):
    if isinstance(iotype, ScalarType):
        return f'"result" {iotype.dtype.to_sql()}'
    column_names = iotype.column_names(name_prefix)
    types = [dtype.to_sql() for _, dtype in iotype.schema]
    sql_params = [f'"{name}" {dtype}' for name, dtype in zip(column_names, types)]
    return SEP.join(sql_params)


# ~~~~~~~~~~~~~~~~~~~~~~~ Helpers ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def recursive_repr(self: object) -> str:
    """Recursively representing an object using its attribute reprs."""
    cls = type(self).__name__
    attrs = self.__dict__
    publicattrs = {
        name: attr for name, attr in attrs.items() if not name.startswith("_")
    }
    attrs_repr = ",".join(name + "=" + repr(attr) for name, attr in publicattrs.items())
    return f"{cls}({attrs_repr})"


def remove_empty_lines(lines: List[str]):
    return [
        remove_trailing_newline(line)
        for line in lines
        if not re.fullmatch(r"^$", line.strip())
    ]


def remove_trailing_newline(string):
    return re.sub(r"\n$", "", string)


def parse_func(func):
    """Get function AST"""
    code = dedent(inspect.getsource(func))
    return ast.parse(code)


def get_func_body_from_ast(tree):
    assert len(tree.body) == 1
    funcdef, *_ = tree.body
    return funcdef.body


def get_return_names_from_body(statements) -> Tuple[str, List[str]]:
    """Returns names of variables in return statement. Assumes that a return
    statement exists and is of type ast.Name or ast.Tuple because the validation is
    supposed to happen before (in validate_func_as_udf)."""
    ret_stmt = next(s for s in statements if isinstance(s, ast.Return))
    if isinstance(ret_stmt.value, ast.Name):
        return ret_stmt.value.id, []  # type: ignore
    elif isinstance(ret_stmt.value, ast.Tuple):
        main_ret = ret_stmt.value.elts[0].id
        sec_rets = [value.id for value in ret_stmt.value.elts[1:]]
        return main_ret, sec_rets
    else:
        raise NotImplementedError


def make_unique_func_name(func) -> str:
    """Creates a unique function name composed of the function name, an
    underscore and the module's name hashed, encoded in base32 and truncated at
    4 chars."""
    full_module_name = func.__module__
    module_name = full_module_name.split(".")[-1]
    hash_ = get_base32_hash(module_name)
    return func.__name__ + "_" + hash_.lower()


def get_base32_hash(string, chars=4):
    hash_ = hashlib.sha256(string.encode("utf-8")).digest()
    hash_ = base64.b32encode(hash_).decode()[:chars]
    return hash_


def get_func_parameter_names(func):
    """Gets the list of parameter names of a function."""
    signature = inspect.signature(func)
    return list(signature.parameters.keys())


def mapping_inverse(mapping):
    """Inverses mapping if it is bijective or raises error if not."""
    if len(set(mapping.keys())) != len(set(mapping.values())):
        raise ValueError(f"Mapping {mapping} cannot be reversed, it is not bijective.")
    return dict(zip(mapping.values(), mapping.keys()))


def compose_mappings(map1, map2):
    """Returns f[x] = map2[map1[x]], or using Haskell's dot notation
    map1 .  map2, if mappings are composable. Raises a ValueError otherwise."""
    if not set(map1.values()) <= set(map2.keys()):
        raise ValueError(f"Mappings are not composable, {map1}, {map2}")
    return {key: map2[val] for key, val in map1.items()}


def merge_mappings_consistently(mappings: List[dict]) -> dict:
    """Merges a list of mappings into a single mapping, raising a ValueError if
    mappings do not coincide."""
    merged = dict()
    for mapping in mappings:
        if not mappings_coincide(merged, mapping):
            raise ValueError(f"Cannot merge inconsistent mappings: {merged}, {mapping}")
        merged.update(mapping)
    return merged


def mappings_coincide(map1: dict, map2: dict) -> bool:
    """Returns True if the image of the intersection of the two mappings is
    the same, False otherwise. In other words, True if the mappings coincide on
    the intersection of their keys."""
    intersection = set(map1.keys()) & set(map2.keys())
    if any(map1[key] != map2[key] for key in intersection):
        return False
    return True


def merge_args_and_kwargs(param_names, args, kwargs):
    """Merges args and kwargs for a given list of parameter names into a single
    dictionary."""
    merged = dict(zip(param_names, args))
    merged.update(kwargs)
    return merged


def get_items_of_type(type_, mapping):
    """Gets items of mapping being instances of a given type."""
    return {key: val for key, val in mapping.items() if isinstance(val, type_)}


def is_any_element_of_type(type_, elements):
    return any(isinstance(elm, type_) for elm in elements)


class UDFBadDefinition(Exception):
    """Raised when an error is detected in the definition of a udf decorated
    function. These checks are made as soon as the function is defined."""


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
    pass


class LoopbackOutputType(OutputType):
    pass


class UDFLoggerType(InputType):
    pass


def udf_logger():
    return UDFLoggerType()


class TableType(ABC):
    @property
    @abstractmethod
    def schema(self):
        raise NotImplementedError

    def column_names(self, prefix=""):
        prefix += "_" if prefix else ""
        return [prefix + name for name, _ in self.schema]


class TensorType(TableType, ParametrizedType, InputType, OutputType):
    def __init__(self, dtype, ndims):
        self.dtype = dt.from_py(dtype) if isinstance(dtype, type) else dtype
        self.ndims = ndims

    @property
    def schema(self):
        dimcolumns = [(f"dim{i}", dt.INT) for i in range(self.ndims)]
        valcolumn = [("val", self.dtype)]
        return dimcolumns + valcolumn


def tensor(dtype, ndims):
    return TensorType(dtype, ndims)


class MergeTensorType(TableType, ParametrizedType, InputType, OutputType):
    def __init__(self, dtype, ndims):
        self.dtype = dt.from_py(dtype) if isinstance(dtype, type) else dtype
        self.ndims = ndims

    @property
    def schema(self):
        nodeid_column = [("node_id", dt.STR)]
        dimcolumns = [(f"dim{i}", dt.INT) for i in range(self.ndims)]
        valcolumn = [("val", self.dtype)]
        return nodeid_column + dimcolumns + valcolumn  # type: ignore


def merge_tensor(dtype, ndims):
    return MergeTensorType(dtype, ndims)


class RelationType(TableType, ParametrizedType, InputType, OutputType):
    def __init__(self, schema):
        if isinstance(schema, TypeVar):
            self._schema = schema
        else:
            self._schema = [
                (name, dt.from_py(dtype) if isinstance(dtype, type) else dtype)
                for name, dtype in schema
            ]

    @property
    def schema(self):
        return self._schema


def relation(schema=None):
    schema = schema or TypeVar("S")
    return RelationType(schema)


class ScalarType(OutputType, ParametrizedType):
    """
    @deprecated
    Use 'RelationType(schema=[("scalar", dtype)])' instead.
    """

    def __init__(self, dtype):
        self.dtype = dt.from_py(dtype) if isinstance(dtype, type) else dtype


def scalar(dtype):
    """
    @deprecated
    Use 'relation(schema=[("scalar", dtype)])' instead.
    """
    return ScalarType(dtype)


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


def transfer():
    return TransferType()


class MergeTransferType(DictType, InputType):
    _data_column_name = "transfer"
    _data_column_type = dt.JSON


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


def secure_transfer(sum_op=False, min_op=False, max_op=False):
    if not sum_op and not min_op and not max_op:
        raise UDFBadDefinition(
            "In a secure_transfer at least one operation should be enabled."
        )
    return SecureTransferType(sum_op, min_op, max_op)


class StateType(DictType, InputType, LoopbackOutputType):
    _data_column_name = "state"
    _data_column_type = dt.BINARY


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
    request_id: str
    udf_name: str

    def __init__(self, request_id, udf_name):
        self.request_id = request_id
        self.udf_name = udf_name


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


# ~~~~~~~~~~~~~~~~~~~~~~~ Type Aliases ~~~~~~~~~~~~~~~~~~~~~~~ #


KnownTypeParams = Union[type, int]
UnknownTypeParams = TypeVar
TypeParamsInference = Dict[UnknownTypeParams, KnownTypeParams]

# ~~~~~~~~~~~~~~~~~~~~~~ UDF AST Nodes ~~~~~~~~~~~~~~~~~~~~~~~ #


class ASTNode(ABC):
    @abstractmethod
    def compile(self) -> str:
        raise NotImplementedError

    __repr__ = recursive_repr

    def __str__(self):
        return self.compile()  # pragma: no cover


class UDFParameter(ASTNode):
    def __init__(self, param_arg, name="") -> None:
        self.name = name
        self.input_type = param_arg.type

    def compile(self) -> str:
        return iotype_to_sql_schema(iotype=self.input_type, name_prefix=self.name)


class UDFReturnType(ASTNode):
    def __init__(self, return_type) -> None:
        self.output_type = return_type

    def compile(self) -> str:
        return get_return_type_template(self.output_type)


class UDFSignature(ASTNode):
    def __init__(
        self,
        udfname: str,  # unused as long as generator returns templates
        table_args: Dict[str, TableArg],
        return_type: OutputType,
    ):
        self.udfname = "$udf_name"
        self.parameter_types = [
            UDFParameter(arg, name)
            for name, arg in table_args.items()
            if isinstance(arg.type, ParametrizedType)
        ]
        self.return_type = UDFReturnType(return_type)

    def compile(self) -> str:
        return LN.join(
            [
                self._format_prototype(),
                RETURNS,
                self.return_type.compile(),
            ]
        )

    def _format_prototype(self):
        return f"{self.udfname}({self._format_parameters()})"

    def _format_parameters(self):
        return SEP.join(param.compile() for param in self.parameter_types)


class UDFHeader(ASTNode):
    def __init__(
        self,
        udfname: str,
        table_args: Dict[str, TableArg],
        return_type: OutputType,
    ):
        self.signature = UDFSignature(udfname, table_args, return_type)

    def compile(self) -> str:
        return LN.join(
            [
                CREATE_OR_REPLACE_FUNCTION,
                self.signature.compile(),
                LANGUAGE_PYTHON,
            ]
        )


class Imports(ASTNode):
    # TODO imports should be dynamic
    def __init__(self, import_pickle, import_json):
        self._import_lines = ["import pandas as pd", "import udfio"]
        if import_pickle:
            pickle_import = "import pickle"
            self._import_lines.append(pickle_import)
        if import_json:
            json_import = "import json"
            self._import_lines.append(json_import)

    def compile(self) -> str:
        return LN.join(self._import_lines)


class TableBuild(ASTNode):
    def __init__(self, arg_name, arg, template):
        self.arg_name = arg_name
        self.arg = arg
        self.template = template

    def compile(self) -> str:
        colnames = self.arg.type.column_names(prefix=self.arg_name)
        return self.template.format(
            varname=self.arg_name,
            colnames=colnames,
            table_name=self.arg.table_name,
        )


class TableBuilds(ASTNode):
    def __init__(self, table_args: Dict[str, TableArg]):
        self.table_builds = [
            TableBuild(arg_name, arg, template=get_table_build_template(arg.type))
            for arg_name, arg in table_args.items()
        ]

    def compile(self) -> str:
        return LN.join([tb.compile() for tb in self.table_builds])


class SMPCBuild(ASTNode):
    def __init__(self, arg_name, arg, template):
        self.arg_name = arg_name
        self.arg = arg
        self.template = template

    def compile(self) -> str:
        return self.template.format(
            varname=self.arg_name,
            template_table_name=self.arg.template_table_name,
            sum_op_values_table_name=self.arg.sum_op_values_table_name,
            min_op_values_table_name=self.arg.min_op_values_table_name,
            max_op_values_table_name=self.arg.max_op_values_table_name,
        )


class SMPCBuilds(ASTNode):
    def __init__(self, smpc_args: Dict[str, SMPCSecureTransferArg]):
        self.smpc_builds = [
            SMPCBuild(arg_name, arg, template=get_smpc_build_template(arg.type))
            for arg_name, arg in smpc_args.items()
        ]

    def compile(self) -> str:
        return LN.join([tb.compile() for tb in self.smpc_builds])


class UDFReturnStatement(ASTNode):
    def __init__(self, return_name, return_type, smpc_used: bool):
        self.return_name = return_name
        self.template = get_main_return_stmt_template(return_type, smpc_used)

    def compile(self) -> str:
        return self.template.format(return_name=self.return_name)


class UDFLoopbackReturnStatements(ASTNode):
    def __init__(self, sec_return_names, sec_return_types, smpc_used):
        self.sec_return_names = sec_return_names
        self.templates = [
            get_secondary_return_stmt_template(sec_return_type, table_name, smpc_used)
            for table_name, sec_return_type in _get_loopback_tables_template_names(
                sec_return_types
            )
        ]

    def compile(self) -> str:
        return LN.join(
            [
                template.format(return_name=return_name)
                for template, return_name in zip(self.templates, self.sec_return_names)
            ]
        )


class LiteralAssignments(ASTNode):
    def __init__(self, literals: Dict[str, LiteralArg]):
        self.literals = literals

    def compile(self) -> str:
        return LN.join(f"{name} = {arg.value}" for name, arg in self.literals.items())


class LoggerAssignment(ASTNode):
    def __init__(self, logger: Optional[Tuple[str, UDFLoggerArg]]):
        self.logger = logger

    def compile(self) -> str:
        if not self.logger:
            return ""
        name, logger_arg = self.logger
        return f"{name} = udfio.get_logger('{logger_arg.udf_name}', '{logger_arg.request_id}')"


class UDFBodyStatements(ASTNode):
    def __init__(self, statements):
        self.returnless_stmts = [
            astor.to_source(stmt)
            for stmt in statements
            if not isinstance(stmt, ast.Return)
        ]

    def compile(self) -> str:
        return LN.join(remove_empty_lines(self.returnless_stmts))


class UDFBody(ASTNode):
    def __init__(
        self,
        table_args: Dict[str, TableArg],
        smpc_args: Dict[str, SMPCSecureTransferArg],
        literal_args: Dict[str, LiteralArg],
        logger_arg: Optional[Tuple[str, UDFLoggerArg]],
        statements: list,
        main_return_name: str,
        main_return_type: OutputType,
        sec_return_names: List[str],
        sec_return_types: List[OutputType],
        smpc_used: bool,
    ):
        self.returnless_stmts = UDFBodyStatements(statements)
        self.loopback_return_stmts = UDFLoopbackReturnStatements(
            sec_return_names=sec_return_names,
            sec_return_types=sec_return_types,
            smpc_used=smpc_used,
        )
        self.return_stmt = UDFReturnStatement(
            main_return_name, main_return_type, smpc_used
        )
        self.table_builds = TableBuilds(table_args)
        self.smpc_builds = SMPCBuilds(smpc_args)
        self.literals = LiteralAssignments(literal_args)
        self.logger = LoggerAssignment(logger_arg)
        all_types = (
            [arg.type for arg in table_args.values()]
            + [main_return_type]
            + sec_return_types
        )

        import_pickle = is_any_element_of_type(StateType, all_types)
        import_json = is_any_element_of_type(
            TransferType, all_types
        ) or is_any_element_of_type(SecureTransferType, all_types)
        self.imports = Imports(
            import_pickle=import_pickle,
            import_json=import_json,
        )

    def compile(self) -> str:
        return LN.join(
            remove_empty_lines(
                [
                    self.imports.compile(),
                    self.table_builds.compile(),
                    self.smpc_builds.compile(),
                    self.literals.compile(),
                    self.logger.compile(),
                    self.returnless_stmts.compile(),
                    self.loopback_return_stmts.compile(),
                    self.return_stmt.compile(),
                ]
            )
        )


class UDFTracebackCatcher(ASTNode):
    try_template = [
        r"import traceback",
        r"try:",
    ]
    except_template = [
        r"except Exception as e:",
        indent(r"offset = 5", prefix=SPC4),
        indent(r"tb = e.__traceback__", prefix=SPC4),
        indent(r"lineno = tb.tb_lineno - offset", prefix=SPC4),
        indent(r"line = ' ' * 4 + __code[lineno]", prefix=SPC4),
        indent(r"linelen = len(__code[lineno])", prefix=SPC4),
        indent(r"underline = ' ' * 4 + '^' * linelen", prefix=SPC4),
        indent(r"tb_lines = traceback.format_tb(tb)", prefix=SPC4),
        indent(r"tb_lines.insert(1, line)", prefix=SPC4),
        indent(r"tb_lines.insert(2, underline)", prefix=SPC4),
        indent(r"tb_lines.append(repr(e))", prefix=SPC4),
        indent(r"tb_formatted = '\n'.join(tb_lines)", prefix=SPC4),
        indent(r"return tb_formatted", prefix=SPC4),
    ]

    def __init__(self, body: UDFBody):
        self.body = body

    def compile(self) -> str:
        body = self.body.compile().splitlines()
        *base_body, _ = body
        recompiled_lines = base_body + ['return "no error"']
        return LN.join(
            [
                f"__code = {base_body}",
                *self.try_template,
                *[indent(line, SPC4) for line in recompiled_lines],
                *self.except_template,
            ]
        )


class UDFDefinition(ASTNode):
    def __init__(
        self,
        funcparts: "FunctionParts",
        table_args: Dict[str, TableArg],
        smpc_args: Dict[str, SMPCSecureTransferArg],
        literal_args: Dict[str, LiteralArg],
        logger_arg: Optional[Tuple[str, UDFLoggerArg]],
        main_output_type: OutputType,
        sec_output_types: List[OutputType],
        smpc_used: bool,
        traceback=False,
    ):
        self.header = UDFHeader(
            udfname=funcparts.qualname,
            table_args=table_args,
            return_type=main_output_type,
        )
        body = UDFBody(
            table_args=table_args,
            smpc_args=smpc_args,
            literal_args=literal_args,
            logger_arg=logger_arg,
            statements=funcparts.body_statements,
            main_return_name=funcparts.main_return_name,
            main_return_type=main_output_type,
            sec_return_names=funcparts.sec_return_names,
            sec_return_types=sec_output_types,
            smpc_used=smpc_used,
        )
        self.body = UDFTracebackCatcher(body) if traceback else body

    def compile(self) -> str:
        return LN.join(
            [
                self.header.compile(),
                BEGIN,
                indent(self.body.compile(), prefix=SPC4),
                END,
            ]
        )


class StarColumn(ASTNode):
    def compile(self, *_, **__):
        return "*"


class ConstColumn(ASTNode):
    def __init__(self, value, alias):
        self.name = str(value)
        self.alias = alias

    def compile(self, use_alias=False):
        return f"{self.name}" + (f' AS "{self.alias}"' if use_alias else "")


class Column(ASTNode):
    def __init__(self, name: str, table: "Table" = None, alias=""):
        self.name = name
        self.table = table
        self.alias = alias

    def compile(self, use_alias=True, use_prefix=True) -> str:
        prefix = (
            self.table.compile(use_alias=False) + "."
            if self.table and use_prefix
            else ""
        )
        postfix = f" AS {self.alias}" if self.alias and use_alias else ""
        return prefix + self.name + postfix

    def __eq__(self, other):
        return ColumnEqualityClause(self, other)

    # XXX naive implementation ignoring operator precedence when more than 2 operands
    # TODO multidispatch dunder methods
    def __add__(self, other):
        if isinstance(other, Column):
            name1 = self.compile(use_alias=False)
            name2 = other.compile(use_alias=False)
            return Column(name=f"{name1} + {name2}")
        if isinstance(other, Number):
            name1 = self.compile(use_alias=False)
            return Column(name=f"{name1} + {other}")
        raise TypeError(f"unsupported operand types for +: Column and {type(other)}")

    def __radd__(self, other):
        if isinstance(other, Number):
            name1 = self.compile(use_alias=False)
            return Column(name=f"{other} + {name1}")
        raise TypeError(f"unsupported operand types for +: {type(other)} and Column")

    def __sub__(self, other):
        if isinstance(other, Column):
            name1 = self.compile(use_alias=False)
            name2 = other.compile(use_alias=False)
            return Column(name=f"{name1} - {name2}")
        if isinstance(other, Number):
            name1 = self.compile(use_alias=False)
            return Column(name=f"{name1} - {other}")
        raise TypeError(f"unsupported operand types for -: Column and {type(other)}")

    def __rsub__(self, other):
        if isinstance(other, Number):
            name1 = self.compile(use_alias=False)
            return Column(name=f"{other} - {name1}")
        raise TypeError(f"unsupported operand types for -: {type(other)} and Column")

    def __mul__(self, other):
        if isinstance(other, Column):
            name1 = self.compile(use_alias=False)
            name2 = other.compile(use_alias=False)
            return Column(name=f"{name1} * {name2}")
        if isinstance(other, Number):
            name1 = self.compile(use_alias=False)
            return Column(name=f"{name1} * {other}")
        raise TypeError(f"unsupported operand types for *: Column and {type(other)}")

    def __rmul__(self, other):
        if isinstance(other, Number):
            name1 = self.compile(use_alias=False)
            return Column(name=f"{other} * {name1}")
        raise TypeError(f"unsupported operand types for *: {type(other)} and Column")

    def __truediv__(self, other):
        if isinstance(other, Column):
            name1 = self.compile(use_alias=False)
            name2 = other.compile(use_alias=False)
            return Column(name=f"{name1} / {name2}")
        if isinstance(other, Number):
            name1 = self.compile(use_alias=False)
            return Column(name=f"{name1} / {other}")
        raise TypeError(f"unsupported operand types for /: Column and {type(other)}")

    def __rtruediv__(self, other):
        if isinstance(other, Number):
            name1 = self.compile(use_alias=False)
            return Column(name=f"{other} / {name1}")
        raise TypeError(f"unsupported operand types for /: {type(other)} and Column")


class Cast(ASTNode):
    def __init__(self, name, type_, alias):
        self.name = name
        self.type_ = type_
        self.alias = alias

    def compile(self, use_alias=False):
        result = f"CAST('{self.name}' AS {self.type_})"
        return result + (f" AS {self.alias}" if use_alias else "")


class ColumnEqualityClause(ASTNode):
    def __init__(self, column1: Column, column2: Column):
        self.column1 = column1
        self.column2 = column2

    def compile(self) -> str:
        col1 = self.column1.compile(use_alias=False, use_prefix=True)
        col2 = self.column2.compile(use_alias=False, use_prefix=True)
        return col1 + "=" + col2


class ScalarFunction(ASTNode):
    def __init__(self, name: str, columns: List[Column], alias=""):
        self.name = name
        self.select_clause = ColumnsClauseParameters(columns, newline=False)
        self.alias = alias

    def compile(self, use_alias=False) -> str:
        postfix = f" AS {self.alias}" if self.alias and use_alias else ""
        return self.name + "(" + self.select_clause.compile() + ")" + postfix


class TableFunction(ASTNode):
    def __init__(self, name: str, subquery: "Select" = None, alias=""):
        self.name = name
        self.subquery = subquery
        self.alias = alias

    # TODO need to change tests to use alias
    def compile(self, use_alias=False) -> str:
        postfix = f" AS {self.alias}" if self.alias and use_alias else ""
        if self.subquery:
            return LN.join(
                [
                    self.name + "((",
                    indent(self.subquery.compile(), prefix=SPC4),
                    "))" + postfix,
                ]
            )
        return self.name + "()" + postfix


class Table(ASTNode):
    def __init__(self, name: str, columns: List[Union[str, Column]], alias=""):
        self.name = name
        if columns and isinstance(columns[0], str):
            self.columns = {colname: Column(colname, self) for colname in columns}
        else:
            self.columns = columns
        self.c = self.columns
        self.alias = alias

    def compile(self, use_alias=False) -> str:
        if use_alias:
            postfix = f" AS {self.alias}" if self.alias else ""
            return self.name + postfix
        return self.alias or self.name


class Select(ASTNode):
    def __init__(
        self,
        columns: List[Union[Column, ScalarFunction]],
        tables: List[Union[Table, TableFunction]],
        where: List[ColumnEqualityClause] = None,
        groupby: List[Column] = None,
        orderby: List[Column] = None,
    ):
        self.select_clause = SelectClause(columns)
        self.from_clause = FromClause(tables)
        self.where_clause = WhereClause(where) if where else None
        self.groupby_clause = GroupbyClause(groupby) if groupby else None
        self.orderby_clause = OrderbyClause(orderby) if orderby else None

    def compile(self) -> str:
        lines = [
            self.select_clause.compile(),
            self.from_clause.compile(),
        ]
        if self.where_clause:
            lines += [
                self.where_clause.compile(),
            ]
        if self.groupby_clause:
            lines += [
                self.groupby_clause.compile(),
            ]
        if self.orderby_clause:
            lines += [
                self.orderby_clause.compile(),
            ]

        return LN.join(lines)


class SelectClause(ASTNode):
    def __init__(self, columns):
        self.columns = ColumnsClauseParameters(columns)

    def compile(self) -> str:
        parameters = self.columns.compile()
        return LN.join([SELECT, indent(parameters, prefix=SPC4)])


def get_distinct_tables(tables):
    distinct_tables = []
    for table in tables:
        table_exists = False
        for d_table in distinct_tables:
            if table.name == d_table.name:
                table_exists = True
                break
        if not table_exists:
            distinct_tables.append(table)
    return distinct_tables


class FromClause(ASTNode):
    def __init__(self, tables, use_alias=True):
        # Remove duplicate tables, sql doesn't accept "FROM test1, test1
        self.tables = get_distinct_tables(tables)
        self.use_alias = use_alias

    def compile(self) -> str:
        parameters = SEPLN.join(
            table.compile(use_alias=self.use_alias) for table in self.tables
        )
        return LN.join([FROM, indent(parameters, prefix=SPC4)])


class WhereClause(ASTNode):
    def __init__(self, clauses):
        self.clauses = clauses

    def compile(self) -> str:
        parameters = ANDLN.join(clause.compile() for clause in self.clauses)
        return LN.join([WHERE, indent(parameters, prefix=SPC4)])


class ColumnsClauseParameters(ASTNode):
    def __init__(self, elements, use_alias=True, newline=True):
        self.elements = elements
        self.use_alias = use_alias
        self.newline = newline

    def compile(self) -> str:
        sep = SEPLN if self.newline else SEP
        return sep.join(
            element.compile(use_alias=self.use_alias) for element in self.elements
        )


class GroupbyClause(ASTNode):
    def __init__(self, columns):
        self.columns = ColumnsClauseParameters(columns, use_alias=False)

    def compile(self) -> str:
        parameters = self.columns.compile()
        return LN.join([GROUP_BY, indent(parameters, prefix=SPC4)])


class OrderbyClause(ASTNode):
    def __init__(self, columns):
        self.columns = columns

    def compile(self) -> str:
        parameters = SEPLN.join(
            column.compile(use_alias=False, use_prefix=False) for column in self.columns
        )
        return LN.join([ORDER_BY, indent(parameters, prefix=SPC4)])


# ~~~~~~~~~~~~~~~~~~~~~~~~~ Decorator ~~~~~~~~~~~~~~~~~~~~~~~~ #


class UDFDecorator:
    registry = {}

    def __call__(self, **kwargs):
        def decorator(func):
            parameter_names = get_func_parameter_names(func)
            validate_decorator_parameter_names(parameter_names, kwargs)
            signature = make_udf_signature(parameter_names, kwargs)
            validate_udf_signature_types(signature)
            validate_udf_return_statement(func)
            funcparts = breakup_function(func, signature)
            validate_udf_table_input_types(funcparts.table_input_types)
            funcname = funcparts.qualname
            self.registry[funcname] = funcparts
            return func

        return decorator


# Singleton pattern
udf = UDFDecorator()
del UDFDecorator


class Signature(NamedTuple):
    parameters: Dict[str, InputType]
    main_return_annotation: OutputType
    sec_return_annotations: List[OutputType]


class FunctionParts(NamedTuple):
    """A function's parts, used in various stages of the udf definition/query
    generation."""

    qualname: str
    body_statements: list
    main_return_name: str
    sec_return_names: List[str]
    table_input_types: Dict[str, TableType]
    literal_input_types: Dict[str, LiteralType]
    logger_param_name: Optional[str]
    main_output_type: OutputType
    sec_output_types: List[OutputType]
    sig: Signature


def validate_decorator_parameter_names(parameter_names, decorator_kwargs):
    """
    Validates:
     1) that decorator parameter names and func kwargs names match.
     2) that "return_type" exists as a decorator parameter.
     3) the udf_logger
    """
    validate_udf_logger(parameter_names, decorator_kwargs)

    if "return_type" not in decorator_kwargs:
        raise UDFBadDefinition("No return_type defined.")
    parameter_names = set(parameter_names)
    decorator_parameter_names = set(decorator_kwargs.keys())
    decorator_parameter_names.remove("return_type")  # not a parameter
    if parameter_names == decorator_parameter_names:
        return

    parameters_not_provided = decorator_parameter_names - parameter_names
    if parameters_not_provided:
        raise UDFBadDefinition(
            f"The parameters: {','.join(parameters_not_provided)} were not provided in the func definition."
        )

    parameters_not_defined_in_dec = parameter_names - decorator_parameter_names
    if parameters_not_defined_in_dec:
        raise UDFBadDefinition(
            f"The parameters: {','.join(parameters_not_defined_in_dec)} were not defined in the decorator."
        )


def validate_udf_logger(parameter_names, decorator_kwargs):
    """
    udf_logger is a special case of a parameter.
    It won't be provided by the user but from the udfgenerator.
    1) Only one input of this type can exist.
    2) It must be the final parameter, so it won't create problems with the positional arguments.
    """
    udf_logger_param_name = None
    for param_name, param_type in decorator_kwargs.items():
        if isinstance(param_type, UDFLoggerType):
            if udf_logger_param_name:
                raise UDFBadDefinition("Only one 'udf_logger' parameter can exist.")
            udf_logger_param_name = param_name

    if not udf_logger_param_name:
        return

    all_parameter_names_but_the_last = parameter_names[:-1]
    if udf_logger_param_name in all_parameter_names_but_the_last:
        raise UDFBadDefinition("'udf_logger' must be the last input parameter.")


def make_udf_signature(parameter_names, decorator_kwargs):
    parameters = {name: decorator_kwargs[name] for name in parameter_names}
    if isinstance(decorator_kwargs["return_type"], List):
        main_return_annotation = decorator_kwargs["return_type"][0]
        sec_return_annotations = decorator_kwargs["return_type"][1:]
    else:
        main_return_annotation = decorator_kwargs["return_type"]
        sec_return_annotations = []
    signature = Signature(
        parameters=parameters,
        main_return_annotation=main_return_annotation,
        sec_return_annotations=sec_return_annotations,
    )
    return signature


def validate_udf_signature_types(funcsig: Signature):
    """Validates that all types used in the udf's type signature, both input
    and output, are subclasses of InputType or OutputType."""
    parameter_types = funcsig.parameters.values()
    if any(not isinstance(input_type, InputType) for input_type in parameter_types):
        raise UDFBadDefinition(
            f"Input types of func are not subclasses of InputType: {parameter_types}."
        )

    main_return = funcsig.main_return_annotation
    if not isinstance(main_return, OutputType):
        raise UDFBadDefinition(
            f"Output type of func is not subclass of OutputType: {main_return}."
        )

    sec_returns = funcsig.sec_return_annotations
    if any(
        not isinstance(output_type, LoopbackOutputType) for output_type in sec_returns
    ):
        raise UDFBadDefinition(
            f"The secondary output types of func are not subclasses of LoopbackOutputType: {sec_returns}."
        )


def validate_udf_return_statement(func):
    """Validates two things concerning the return statement of a udf.
    1) that there is one and
    2) that it is of the simple `return foo, bar` form, as no
    expressions are allowed in udf return statements."""
    tree = parse_func(func)
    statements = get_func_body_from_ast(tree)
    try:
        ret_stmt = next(s for s in statements if isinstance(s, ast.Return))
    except StopIteration as stop_iter:
        raise UDFBadDefinition(f"Return statement not found in {func}.") from stop_iter
    if not isinstance(ret_stmt.value, ast.Name) and not isinstance(
        ret_stmt.value, ast.Tuple
    ):
        raise UDFBadDefinition(
            f"Expression in return statement in {func}."
            "Assign expression to variable/s and return it/them."
        )


def breakup_function(func, funcsig) -> FunctionParts:
    """Breaks up a function into smaller parts, which will be used during
    the udf translation process."""
    qualname = make_unique_func_name(func)
    tree = parse_func(func)
    body_statements = get_func_body_from_ast(tree)
    main_return_name, sec_return_names = get_return_names_from_body(body_statements)
    table_input_types = {
        name: input_type
        for name, input_type in funcsig.parameters.items()
        if isinstance(input_type, TableType)
    }
    literal_input_types = {
        name: input_type
        for name, input_type in funcsig.parameters.items()
        if isinstance(input_type, LiteralType)
    }

    logger_param_name = None
    for name, input_type in funcsig.parameters.items():
        if isinstance(input_type, UDFLoggerType):
            logger_param_name = name
            break  # Only one logger is allowed

    main_output_type = funcsig.main_return_annotation
    sec_output_types = funcsig.sec_return_annotations
    return FunctionParts(
        qualname=qualname,
        body_statements=body_statements,
        main_return_name=main_return_name,
        sec_return_names=sec_return_names,
        table_input_types=table_input_types,
        literal_input_types=literal_input_types,
        logger_param_name=logger_param_name,
        main_output_type=main_output_type,
        sec_output_types=sec_output_types,
        sig=funcsig,
    )


def validate_udf_table_input_types(table_input_types):
    tensors = get_items_of_type(TensorType, table_input_types)
    relations = get_items_of_type(RelationType, table_input_types)
    if tensors and relations:
        raise UDFBadDefinition("Cannot pass both tensors and relations to udf.")


# ~~~~~~~~~~~~~~~~~ Module Public Function ~~~~~~~~~~~~~~~ #


LiteralValue = Union[Number, numpy.ndarray]
UDFGenArgument = Union[TableInfo, LiteralValue, SMPCTablesInfo]


class UDFBadCall(Exception):
    """Raised when something is wrong with the arguments passed to the udf
    generator."""


def generate_udf_queries(
    request_id: str,
    func_name: str,
    positional_args: List[UDFGenArgument],
    keyword_args: Dict[str, UDFGenArgument],
    smpc_used: bool,
    traceback=False,
) -> UDFGenExecutionQueries:
    """
    Parameters
    ----------
    request_id: An identifier for logging purposes
    func_name: The name of the udf to run
    positional_args: Positional arguments
    keyword_args: Keyword arguments
    smpc_used: Is SMPC used in the computations?
    traceback: Run the udf with traceback enabled to get logs

    Returns
    -------
    a UDFExecutionQueries object.

    """
    udf_posargs, udf_kwargs = convert_udfgenargs_to_udfargs(
        positional_args,
        keyword_args,
        smpc_used,
    )

    if func_name in TENSOR_OP_NAMES:
        if udf_kwargs:
            raise UDFBadCall("Keyword args are not supported for tensor operations.")
        udf_select = get_sql_tensor_operation_select_query(udf_posargs, func_name)
        output_type = get_output_type_for_sql_tensor_operation(func_name, udf_posargs)
        udf_outputs = get_udf_outputs(output_type, None, False)
        udf_execution_query = get_udf_execution_template(udf_select)
        return UDFGenExecutionQueries(
            udf_results=udf_outputs,
            udf_select_query=udf_execution_query,
        )

    if func_name == "create_dummy_encoded_design_matrix":
        return get_create_dummy_encoded_design_matrix_execution_queries(keyword_args)

    return get_udf_templates_using_udfregistry(
        request_id=request_id,
        funcname=func_name,
        posargs=udf_posargs,
        keywordargs=udf_kwargs,
        udfregistry=udf.registry,
        smpc_used=smpc_used,
        traceback=traceback,
    )


def convert_udfgenargs_to_udfargs(udfgen_posargs, udfgen_kwargs, smpc_used=False):
    """
    Converts the udfgen arguments coming from the NODE to the
    "internal" arguments, used only in this udfgenerator module.

    The internal arguments need to be a child class of InputType.
    """
    udf_posargs = [
        convert_udfgenarg_to_udfarg(arg, smpc_used) for arg in udfgen_posargs
    ]
    udf_keywordargs = {
        name: convert_udfgenarg_to_udfarg(arg, smpc_used)
        for name, arg in udfgen_kwargs.items()
    }
    return udf_posargs, udf_keywordargs


def convert_udfgenarg_to_udfarg(udfgen_arg, smpc_used) -> UDFArgument:
    if isinstance(udfgen_arg, SMPCTablesInfo):
        if not smpc_used:
            raise UDFBadCall("SMPC is not used, so SMPCTablesInfo cannot be used.")
        return convert_smpc_udf_input_to_udf_arg(udfgen_arg)
    if isinstance(udfgen_arg, TableInfo):
        return convert_table_info_to_table_arg(udfgen_arg, smpc_used)
    return LiteralArg(value=udfgen_arg)


def convert_smpc_udf_input_to_udf_arg(smpc_udf_input: SMPCTablesInfo):
    sum_op_table_name = None
    min_op_table_name = None
    max_op_table_name = None
    if smpc_udf_input.sum_op_values:
        sum_op_table_name = smpc_udf_input.sum_op_values.name
    if smpc_udf_input.min_op_values:
        min_op_table_name = smpc_udf_input.min_op_values.name
    if smpc_udf_input.max_op_values:
        max_op_table_name = smpc_udf_input.max_op_values.name
    return SMPCSecureTransferArg(
        template_table_name=smpc_udf_input.template.name,
        sum_op_values_table_name=sum_op_table_name,
        min_op_values_table_name=min_op_table_name,
        max_op_values_table_name=max_op_table_name,
    )


def convert_table_info_to_table_arg(table_info, smpc_used):
    if is_transfertype_schema(table_info.schema_.columns):
        return TransferArg(table_name=table_info.name)
    if is_secure_transfer_type_schema(table_info.schema_.columns):
        if smpc_used:
            raise UDFBadCall(
                "When smpc is used SecureTransferArg should not be used, only SMPCSecureTransferArg. "
            )
        return SecureTransferArg(table_name=table_info.name)
    if is_statetype_schema(table_info.schema_.columns):
        if table_info.type_ == DBTableType.REMOTE:
            raise UDFBadCall("Usage of state is only allowed on local tables.")
        return StateArg(table_name=table_info.name)
    if is_tensor_schema(table_info.schema_.columns):
        ndims = (
            len(table_info.schema_.columns) - 2
        )  # TODO avoid this using kinds of TableInfo
        valcol = next(col for col in table_info.schema_.columns if col.name == "val")
        dtype = valcol.dtype
        return TensorArg(table_name=table_info.name, dtype=dtype, ndims=ndims)
    relation_schema = convert_table_schema_to_relation_schema(
        table_info.schema_.columns
    )
    return RelationArg(table_name=table_info.name, schema=relation_schema)


# TODO table kinds must become known in Controller, who should send the
# appropriate kind, avoiding heuristics like below
def is_tensor_schema(schema):
    colnames = [col.name for col in schema]
    if "val" in colnames and any(cname.startswith("dim") for cname in colnames):
        return True
    return False


def is_transfertype_schema(schema):
    schema = [(col.name, col.dtype) for col in schema]
    return all(column in schema for column in TransferType().schema)


def is_secure_transfer_type_schema(schema):
    schema = [(col.name, col.dtype) for col in schema]
    return all(column in schema for column in SecureTransferType().schema)


def is_statetype_schema(schema):
    schema = [(col.name, col.dtype) for col in schema]
    return all(column in schema for column in StateType().schema)


def convert_table_schema_to_relation_schema(table_schema):
    return [(c.name, c.dtype) for c in table_schema if c.name != ROWID]


# <--


def get_udf_templates_using_udfregistry(
    request_id: str,
    funcname: str,
    posargs: List[UDFArgument],
    keywordargs: Dict[str, UDFArgument],
    udfregistry: dict,
    smpc_used: bool,
    traceback=False,
) -> UDFGenExecutionQueries:
    funcparts = get_funcparts_from_udf_registry(funcname, udfregistry)
    udf_args = get_udf_args(request_id, funcparts, posargs, keywordargs)
    udf_args = resolve_merge_table_args(
        udf_args=udf_args,
        expected_table_types=funcparts.table_input_types,
    )
    validate_arg_names(udf_args=udf_args, parameters=funcparts.sig.parameters)
    validate_arg_types(
        udf_args=udf_args,
        expected_tables_types=funcparts.table_input_types,
        expected_literal_types=funcparts.literal_input_types,
    )
    main_output_type, sec_output_types = get_output_types(
        funcparts, udf_args, traceback
    )
    udf_outputs = get_udf_outputs(
        main_output_type=main_output_type,
        sec_output_types=sec_output_types,
        smpc_used=smpc_used,
    )

    udf_definition = get_udf_definition_template(
        funcparts=funcparts,
        input_args=udf_args,
        main_output_type=main_output_type,
        sec_output_types=sec_output_types,
        smpc_used=smpc_used,
        traceback=traceback,
    )

    table_args = get_items_of_type(TableArg, mapping=udf_args)
    udf_select = get_udf_select_template(
        ScalarType(str) if traceback else main_output_type,
        table_args,
    )
    udf_execution = get_udf_execution_template(udf_select)

    return UDFGenExecutionQueries(
        udf_results=udf_outputs,
        udf_definition_query=udf_definition,
        udf_select_query=udf_execution,
    )


def get_output_type_for_sql_tensor_operation(funcname, posargs):
    """Computes the output type for SQL tensor operations. The output is
    allways of type TensorType with float dtype but the dimensions must be
    determined.  SQL tensor operation suport only matmul and elementwise
    operations, for now. Hence the output dimensions are either matmul's result
    or equal to the first argument's dimensions."""
    if funcname == TensorBinaryOp.MATMUL.name:
        out_ndims = sum(tensor_arg.ndims for tensor_arg in posargs) - 2
    else:
        a_tensor = next(arg for arg in posargs if isinstance(arg, TensorArg))
        out_ndims = a_tensor.ndims
    output_type = tensor(dtype=float, ndims=out_ndims)
    return output_type


def get_udf_args(request_id, funcparts, posargs, keywordargs) -> Dict[str, UDFArgument]:
    udf_args = merge_args_and_kwargs(
        param_names=funcparts.sig.parameters.keys(),
        args=posargs,
        kwargs=keywordargs,
    )

    # Check logger_param_name argument is not given and if not, create it.
    if funcparts.logger_param_name:
        if funcparts.logger_param_name in udf_args.keys():
            raise UDFBadCall(
                f"No argument should be provided for 'UDFLoggerType' parameter: '{funcparts.logger_param_name}'"
            )
        udf_args[funcparts.logger_param_name] = UDFLoggerArg(
            request_id=request_id,
            udf_name=funcparts.qualname,
        )
    return udf_args


def resolve_merge_table_args(
    udf_args: Dict[str, UDFArgument],
    expected_table_types: Dict[str, TableType],
) -> Dict[str, UDFArgument]:
    """MergeTableTypes have the same schema as the tables that they are merging.
    The UDFArgument always contains the initial table type and must be resolved
    to a MergeTableType, if needed, based on the function parts."""

    def is_merge_tensor(arg, argname, exp_types):
        is_tensor = isinstance(arg, TensorArg)
        return is_tensor and isinstance(exp_types[argname], MergeTensorType)

    def is_merge_transfer(arg, argname, exp_types):
        is_transfer = isinstance(arg, TransferArg)
        return is_transfer and isinstance(exp_types[argname], MergeTransferType)

    udf_args = deepcopy(udf_args)
    for argname, arg in udf_args.items():
        if is_merge_tensor(arg, argname, expected_table_types):
            tensor_type = arg.type
            arg.type = merge_tensor(dtype=tensor_type.dtype, ndims=tensor_type.ndims)
        if is_merge_transfer(arg, argname, expected_table_types):
            udf_args[argname].type = merge_transfer()

    return udf_args


def get_funcparts_from_udf_registry(funcname: str, udfregistry: dict) -> FunctionParts:
    if funcname not in udfregistry:
        raise UDFBadCall(f"{funcname} cannot be found in udf registry.")
    return udfregistry[funcname]


def validate_arg_names(
    udf_args: Dict[str, UDFArgument],
    parameters: Dict[str, InputType],
) -> None:
    """Validates that the names of the udf arguments are the expected ones,
    based on the udf's formal parameters."""
    if udf_args.keys() != parameters.keys():
        raise UDFBadCall(
            f"UDF argument names do not match UDF parameter names: "
            f"{udf_args.keys()}, {parameters.keys()}."
        )


def validate_arg_types(
    udf_args: Dict[str, UDFArgument],
    expected_tables_types: Dict[str, TableType],
    expected_literal_types: Dict[str, LiteralType],
) -> None:
    """Validates that the types of the udf arguments are the expected ones,
    based on the udf's formal parameter types."""
    table_args = get_items_of_type(TableArg, udf_args)
    smpc_args = get_items_of_type(SMPCSecureTransferArg, udf_args)
    literal_args = get_items_of_type(LiteralArg, udf_args)
    for argname, arg in table_args.items():
        if not isinstance(arg.type, type(expected_tables_types[argname])):
            raise UDFBadCall(
                f"Argument {argname} should be of type {expected_tables_types[argname]}. "
                f"Type provided: {arg.type}"
            )
    for argname, arg in smpc_args.items():
        if not isinstance(arg.type, type(expected_tables_types[argname])):
            raise UDFBadCall(
                f"Argument {argname} should be of type {expected_tables_types[argname]}. "
                f"Type provided: {arg.type}"
            )
    for argname, arg in literal_args.items():
        if not isinstance(arg.type, type(expected_literal_types[argname])):
            raise UDFBadCall(
                f"Argument {argname} should be of type {expected_tables_types[argname]}. "
                f"Type provided: {arg.type}"
            )


def copy_types_from_udfargs(udfargs: Dict[str, UDFArgument]) -> Dict[str, InputType]:
    return {name: deepcopy(arg.type) for name, arg in udfargs.items()}


# ~~~~~~~~~~~~~~~ UDF Definition Translator ~~~~~~~~~~~~~~ #


def get_udf_definition_template(
    funcparts: FunctionParts,
    input_args: Dict[str, UDFArgument],
    main_output_type: OutputType,
    sec_output_types: List[OutputType],
    smpc_used: bool,
    traceback=False,
) -> Template:
    table_args: Dict[str, TableArg] = get_items_of_type(TableArg, mapping=input_args)
    smpc_args: Dict[str, SMPCSecureTransferArg] = get_items_of_type(
        SMPCSecureTransferArg, mapping=input_args
    )
    literal_args: Dict[str, LiteralArg] = get_items_of_type(
        LiteralArg, mapping=input_args
    )
    logger_arg: Optional[str, UDFLoggerArg] = None
    logger_param = funcparts.logger_param_name
    if logger_param:
        logger_arg = (logger_param, input_args[logger_param])

    verify_declared_and_passed_param_types_match(
        funcparts.table_input_types, table_args
    )
    udf_definition = UDFDefinition(
        funcparts=funcparts,
        table_args=table_args,
        smpc_args=smpc_args,
        literal_args=literal_args,
        logger_arg=logger_arg,
        main_output_type=main_output_type,
        sec_output_types=sec_output_types,
        smpc_used=smpc_used,
        traceback=traceback,
    )
    return Template(udf_definition.compile())


def get_output_types(
    funcparts, udf_args, traceback
) -> Tuple[OutputType, List[LoopbackOutputType]]:
    """Computes the UDF output type. If `traceback` is true the type is str
    since the traceback will be returned as a string. If the output type is
    generic its type parameters must be inferred from the passed input types.
    Otherwise, the declared output type is returned."""
    input_types = copy_types_from_udfargs(udf_args)

    if traceback:
        return scalar(str), []
    if (
        isinstance(funcparts.main_output_type, ParametrizedType)
        and funcparts.main_output_type.is_generic
    ):
        param_table_types = get_items_of_type(TableType, mapping=input_types)
        return (
            infer_output_type(
                passed_input_types=param_table_types,
                declared_input_types=funcparts.table_input_types,
                declared_output_type=funcparts.main_output_type,
            ),
            [],
        )
    return funcparts.main_output_type, funcparts.sec_output_types


def verify_declared_and_passed_param_types_match(
    declared_types: Dict[str, TableType],
    passed_args: Dict[str, TableArg],
) -> None:
    passed_param_args = {
        name: arg
        for name, arg in passed_args.items()
        if isinstance(arg.type, ParametrizedType)
    }
    for argname, arg in passed_param_args.items():
        known_params = declared_types[argname].known_typeparams
        verify_declared_typeparams_match_passed_type(known_params, arg.type)


def verify_declared_typeparams_match_passed_type(
    known_typeparams: Dict[str, KnownTypeParams],
    passed_type: ParametrizedType,
) -> None:
    for name, param in known_typeparams.items():
        if not hasattr(passed_type, name):
            raise UDFBadCall(f"{passed_type} has no typeparam {name}.")
        if getattr(passed_type, name) != param:
            raise UDFBadCall(
                "InputType's known typeparams do not match typeparams passed "
                f"in {passed_type}: {param}, {getattr(passed_type, name)}."
            )


def infer_output_type(
    passed_input_types: Dict[str, ParametrizedType],
    declared_input_types: Dict[str, ParametrizedType],
    declared_output_type: ParametrizedType,
) -> Union[ParametrizedType, ScalarType]:
    inferred_input_typeparams = infer_unknown_input_typeparams(
        declared_input_types,
        passed_input_types,
    )
    known_output_typeparams = dict(**declared_output_type.known_typeparams)
    inferred_output_typeparams = compose_mappings(
        declared_output_type.unknown_typeparams,
        inferred_input_typeparams,
    )
    known_output_typeparams.update(inferred_output_typeparams)
    inferred_output_type = type(declared_output_type)(**known_output_typeparams)
    return inferred_output_type


def infer_unknown_input_typeparams(
    declared_input_types: Dict[str, ParametrizedType],
    passed_input_types: Dict[str, ParametrizedType],
) -> TypeParamsInference:
    typeparams_inference_mappings = [
        map_unknown_to_known_typeparams(
            input_type.unknown_typeparams,
            passed_input_types[name].known_typeparams,
        )
        for name, input_type in declared_input_types.items()
        if input_type.is_generic
    ]
    distinct_inferred_typeparams = merge_mappings_consistently(
        typeparams_inference_mappings
    )
    return distinct_inferred_typeparams


def map_unknown_to_known_typeparams(
    unknown_params: Dict[str, UnknownTypeParams],
    known_params: Dict[str, KnownTypeParams],
) -> TypeParamsInference:
    return compose_mappings(mapping_inverse(unknown_params), known_params)


# ~~~~~~~~~~~~~~ UDF SELECT Query Generator ~~~~~~~~~~~~~~ #


def get_udf_select_template(output_type: OutputType, table_args: Dict[str, TableArg]):
    tensors = get_table_ast_nodes_from_table_args(
        table_args,
        arg_type=(TensorArg),
    )
    relations = get_table_ast_nodes_from_table_args(table_args, arg_type=RelationArg)
    tables = tensors or relations
    columns = [column for table in tables for column in table.columns.values()]
    where_clause = get_where_clause_for_tensors(tensors) if tensors else None
    where_clause = (
        get_where_clause_for_relations(relations) if relations else where_clause
    )
    if isinstance(output_type, ScalarType):
        func = ScalarFunction(name="$udf_name", columns=columns)
        select_stmt = Select([func], tables, where_clause)
        return select_stmt.compile()
    if isinstance(output_type, TableType):
        subquery = Select(columns, tables, where_clause) if tables else None
        func = TableFunction(name="$udf_name", subquery=subquery)
        select_stmt = Select([nodeid_column(), StarColumn()], [func])
        return select_stmt.compile()
    raise TypeError(f"Got {output_type} as output. Expected ScalarType or TableType")


def get_table_ast_nodes_from_table_args(table_args, arg_type):
    return [
        Table(name=table.table_name, columns=table.column_names())
        for table in get_items_of_type(arg_type, table_args).values()
    ]


def get_where_clause_for_tensors(tensors):
    head_tensor, *tail_tensors = tensors
    where_clause = [
        head_tensor.c[colname] == table.c[colname]
        for table in tail_tensors
        for colname in head_tensor.columns.keys()
        if colname.startswith("dim")
    ]
    return where_clause


def get_where_clause_for_relations(relations):
    if len(relations) == 1:
        return None

    head_relation, *tail_relations = relations
    where_clause = [
        ColumnEqualityClause(
            column1=Column(
                name="row_id",
                table=head_relation,
            ),
            column2=Column(
                name="row_id",
                table=relation,
            ),
        )
        for relation in tail_relations
    ]
    return where_clause


def convert_table_arg_to_table_ast_node(table_arg, alias=None):
    return Table(
        name=table_arg.table_name,
        columns=table_arg.column_names(),
        alias=alias,
    )


def nodeid_column():
    return Cast(name="$node_id", type_=dt.STR.to_sql(), alias="node_id")


# ~~~~~~~~~~~~~~~~~ CREATE TABLE and INSERT query generator ~~~~~~~~~ #
def _get_main_table_template_name() -> str:
    return "main_output_table_name"


def _get_loopback_tables_template_names(
    output_types: List[LoopbackOutputType],
) -> List[Tuple[str, LoopbackOutputType]]:
    """
    Receives a list of LoopbackOutputType and returns a list of the same LoopbackOutputType
    with their corresponding table_tmpl_name.
    """
    return [
        (f"loopback_table_name_{pos}", output_type)
        for pos, output_type in enumerate(output_types)
    ]


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


def _create_table_udf_output(
    output_type: OutputType,
    table_name: str,
    nodeid=True,
) -> UDFGenResult:
    drop_table = DROP_TABLE_IF_EXISTS + " $" + table_name + SCOLON
    if isinstance(output_type, ScalarType) or not nodeid:
        output_schema = iotype_to_sql_schema(output_type)
    else:
        output_schema = f'"node_id" {dt.STR.to_sql()},' + iotype_to_sql_schema(
            output_type
        )
    create_table = CREATE_TABLE + " $" + table_name + f"({output_schema})" + SCOLON
    return TableUDFGenResult(
        tablename_placeholder=table_name,
        drop_query=Template(drop_table),
        create_query=Template(create_table),
    )


def _create_smpc_udf_output(output_type: SecureTransferType, table_name_prefix: str):
    (
        template_tmpl,
        sum_op_tmpl,
        min_op_tmpl,
        max_op_tmpl,
    ) = _get_smpc_table_template_names(table_name_prefix)
    template = _create_table_udf_output(output_type, template_tmpl)
    sum_op = None
    min_op = None
    max_op = None
    if output_type.sum_op:
        sum_op = _create_table_udf_output(output_type, sum_op_tmpl)
    if output_type.min_op:
        min_op = _create_table_udf_output(output_type, min_op_tmpl)
    if output_type.max_op:
        max_op = _create_table_udf_output(output_type, max_op_tmpl)
    return SMPCUDFGenResult(
        template=template,
        sum_op_values=sum_op,
        min_op_values=min_op,
        max_op_values=max_op,
    )


def _create_udf_output(
    output_type: OutputType,
    table_name: str,
    smpc_used: bool,
    nodeid=True,
) -> UDFGenResult:
    if isinstance(output_type, SecureTransferType) and smpc_used:
        return _create_smpc_udf_output(output_type, table_name)
    else:
        return _create_table_udf_output(output_type, table_name, nodeid)


def get_udf_outputs(
    main_output_type: OutputType,
    sec_output_types: List[LoopbackOutputType],
    smpc_used: bool,
    nodeid=True,
) -> List[UDFGenResult]:
    table_name = _get_main_table_template_name()
    udf_outputs = [_create_udf_output(main_output_type, table_name, smpc_used, nodeid)]

    if sec_output_types:
        for table_name, sec_output_type in _get_loopback_tables_template_names(
            sec_output_types
        ):
            udf_outputs.append(
                _create_udf_output(sec_output_type, table_name, smpc_used)
            )

    return udf_outputs


def get_udf_execution_template(udf_select_query) -> Template:
    table_name = _get_main_table_template_name()
    udf_execution = f"INSERT INTO ${table_name}\n" + udf_select_query + SCOLON
    return Template(udf_execution)


# ~~~~~~~~~~~~~~ Pure SQL Tensor Operations ~~~~~~~~~~~~~~ #

import operator


class TensorBinaryOp(Enum):
    ADD = operator.add
    SUB = operator.sub
    MUL = operator.mul
    DIV = operator.truediv
    MATMUL = operator.matmul


class TensorUnaryOp(Enum):
    TRANSPOSE = 0


TENSOR_OP_NAMES = TensorBinaryOp.__members__.keys() | TensorUnaryOp.__members__.keys()


def get_sql_tensor_operation_select_query(udf_posargs, func_name):
    if func_name == TensorUnaryOp.TRANSPOSE.name:
        assert len(udf_posargs) == 1
        matrix, *_ = udf_posargs
        assert matrix.ndims == 2
        return get_matrix_transpose_template(matrix)
    if len(udf_posargs) == 2:
        operand1, operand2 = udf_posargs
        return get_tensor_binary_op_template(
            operand1, operand2, getattr(TensorBinaryOp, func_name)
        )
    raise NotImplementedError


def get_tensor_binary_op_template(
    operand_0: TensorArg,
    operand_1: TensorArg,
    operator: TensorBinaryOp,
):
    if operator is TensorBinaryOp.MATMUL:
        return get_tensor_matmul_template(operand_0, operand_1)
    return get_tensor_elementwise_binary_op_template(operand_0, operand_1, operator)


def get_tensor_elementwise_binary_op_template(
    operand_0: TensorArg,
    operand_1: TensorArg,
    operator: TensorBinaryOp,
):
    if isinstance(operand_0, TensorArg) and isinstance(operand_1, TensorArg):
        return get_tensor_tensor_elementwise_op_template(operand_0, operand_1, operator)
    if isinstance(operand_0, LiteralArg) ^ isinstance(operand_1, LiteralArg):
        return get_tensor_number_binary_op_template(operand_0, operand_1, operator)
    raise NotImplementedError


def get_tensor_tensor_elementwise_op_template(tensor0, tensor1, operator):
    if tensor0.ndims != tensor1.ndims:
        raise NotImplementedError(
            "Cannot perform elementwise operation if the operand "
            f"dimensions are different: {tensor0.ndims}, {tensor1.ndims}"
        )
    table0 = convert_table_arg_to_table_ast_node(tensor0, alias="tensor_0")
    table1 = convert_table_arg_to_table_ast_node(tensor1, alias="tensor_1")

    columns = get_columns_for_tensor_tensor_binary_op(table0, table1, operator)
    where = get_where_params_for_tensor_tensor_binary_op(table0, table1)

    select_stmt = Select(
        columns=columns,
        tables=[table0, table1],
        where=where,
    )
    return select_stmt.compile()


def get_columns_for_tensor_tensor_binary_op(table0, table1, operator):
    columns = [
        column for name, column in table0.columns.items() if name.startswith("dim")
    ]
    for column in columns:
        column.alias = column.name
    valcolumn = operator.value(table0.c["val"], table1.c["val"])
    valcolumn.alias = "val"
    columns = [nodeid_column()] + columns
    columns += [valcolumn]
    return columns


def get_where_params_for_tensor_tensor_binary_op(table0, table1):
    where = [
        table0.c[colname] == table1.c[colname]
        for colname in table0.columns
        if colname.startswith("dim")
    ]
    return where


def get_tensor_number_binary_op_template(operand_0, operand_1, operator):
    if isinstance(operand_0, LiteralArg):
        number = operand_0.value
        table = convert_table_arg_to_table_ast_node(operand_1, alias="tensor_0")
        valcolumn = operator.value(number, table.c["val"])
        valcolumn.alias = "val"
    else:
        number = operand_1.value
        table = convert_table_arg_to_table_ast_node(operand_0, alias="tensor_0")
        valcolumn = operator.value(table.c["val"], number)
        valcolumn.alias = "val"
    columns = get_columns_for_tensor_number_binary_op(table, valcolumn)
    select_stmt = Select(columns, tables=[table])

    return select_stmt.compile()


def get_columns_for_tensor_number_binary_op(table, valcolumn):
    columns = [
        column for name, column in table.columns.items() if name.startswith("dim")
    ]
    for column in columns:
        column.alias = column.name
    columns = [nodeid_column()] + columns
    columns += [valcolumn]
    return columns


def get_tensor_matmul_template(tensor0: TensorArg, tensor1: TensorArg):
    ndims0, ndims1 = tensor0.ndims, tensor1.ndims
    if ndims0 not in (1, 2) or ndims1 not in (1, 2):
        raise NotImplementedError(
            "Cannot multiply tensors of dimension greated than 2."
        )
    table0 = convert_table_arg_to_table_ast_node(tensor0)
    table1 = convert_table_arg_to_table_ast_node(tensor1)
    table0.alias, table1.alias = "tensor_0", "tensor_1"
    ndims0, ndims1 = tensor0.ndims, tensor1.ndims

    columns = get_columns_for_tensor_matmul(ndims0, ndims1, table0, table1)
    where = get_where_params_for_tensor_matmul(ndims0, table0, table1)
    groupby = get_groupby_params_for_tensor_matmul(ndims0, ndims1, table0, table1)
    orderby = get_orderby_params_for_tensor_matmul(ndims0, ndims1)

    tables = [table0, table1]
    select_stmt = Select(columns, tables, where, groupby, orderby)
    return select_stmt.compile()


def get_columns_for_tensor_matmul(ndims0, ndims1, table0, table1):
    """After a contraction, the resulting tensor will contain all indices
    which are not contracted. The select part involves all those indices."""
    tables = [table0, table1]
    remaining_dims = compute_remaining_dimensions_after_contraction(ndims0, ndims1)
    # The (1, 2) case is an exception to the rule, where a transposition is
    # also required on the result. This is due to the fact that vec @ mat is,
    # strictly speaking, actually vec.T @ mat. Numpy, however, allows the
    # operation without requiring a transposition on the first operand and I
    # try to follow numpy's behaviour as much as possible.
    if (ndims0, ndims1) == (1, 2):
        tables[1].c["dim1"].alias = "dim0"
        columns = [tables[1].c["dim1"]]
    else:
        for i in range(remaining_dims):
            tables[i].c[f"dim{i}"].alias = f"dim{i}"
        columns = [tables[i].c[f"dim{i}"] for i in range(remaining_dims)]

    columns = [nodeid_column()] + columns
    prods_column = table0.c["val"] * table1.c["val"]
    sum_of_prods = ScalarFunction("SUM", [prods_column], alias="val")
    columns += [sum_of_prods]
    return columns


def get_where_params_for_tensor_matmul(ndims0, table_0, table_1):
    """The where clause enforces equality on contracted indices."""
    ci0, ci1 = compute_contracted_indices(ndims0)
    where_params = [table_0.c[f"dim{ci0}"] == table_1.c[f"dim{ci1}"]]
    return where_params


def get_groupby_params_for_tensor_matmul(ndims0, ndims1, table0, table1):
    """Similar to the select case, indices which are not contracted are grouped
    together in order to compute the sum of products over all combinations of
    non contracted indices."""
    ri0, ri1 = compute_remaining_indices_after_contraction(ndims0, ndims1)
    groupby_params = [table0.c[f"dim{i}"] for i in ri0]
    groupby_params += [table1.c[f"dim{i}"] for i in ri1]
    return groupby_params


def get_orderby_params_for_tensor_matmul(ndims0, ndims1):
    """The order by clause is not required for correctness but for readability.
    All indices in the resulting table are ordered by ascending dimension."""
    remaining_dims = compute_remaining_dimensions_after_contraction(ndims0, ndims1)
    orderby_params = [Column(f"dim{i}") for i in range(remaining_dims)]
    return orderby_params


def compute_contracted_indices(ndims0):
    """During matrix multiplication the last index of the first tensor gets
    contracted with the first index of the second tensor."""
    return ndims0 - 1, 0


def compute_remaining_dimensions_after_contraction(ndims0, ndims1):
    return (ndims0 - 1) + (ndims1 - 1)


def compute_remaining_indices_after_contraction(ndims0, ndims1):
    *indices0, _ = range(ndims0)
    _, *indices1 = range(ndims1)
    return indices0, indices1


def get_matrix_transpose_template(matrix):
    table = Table(
        name=matrix.table_name,
        columns=matrix.column_names(),
        alias="tensor_0",
    )
    table.c["dim0"].alias = "dim1"
    table.c["dim1"].alias = "dim0"
    table.c["val"].alias = "val"
    select_stmt = Select(
        [nodeid_column(), table.c["dim1"], table.c["dim0"], table.c["val"]],
        [table],
    )
    return select_stmt.compile()


# ~~~~~~~~~~~~~~ SQL special queries ~~~~~~~~~~~~~~ #
def get_create_dummy_encoded_design_matrix_execution_queries(keyword_args):
    dm_table = get_dummy_encoded_design_matrix_table(keyword_args)
    udf_select = get_dummy_encoded_design_matrix_select_stmt(dm_table)
    output_schema = get_dummy_encoded_design_matrix_schema(dm_table)
    output_type = relation(schema=output_schema)
    udf_outputs = get_udf_outputs(
        output_type,
        sec_output_types=None,
        smpc_used=False,
        nodeid=False,
    )
    udf_execution_query = get_udf_execution_template(udf_select)
    return UDFGenExecutionQueries(
        udf_results=udf_outputs,
        udf_select_query=udf_execution_query,
    )


def get_dummy_encoded_design_matrix_table(keyword_args):
    enums = keyword_args["enums"]
    numerical_vars = keyword_args["numerical_vars"]
    intercept = keyword_args["intercept"]
    table_name = keyword_args["x"].name
    rowid_column = [Column(name="row_id")]
    intercept_column = [ConstColumn(value=1, alias="intercept")] if intercept else []
    numerical_columns = [Column(name=varname) for varname in numerical_vars]
    dummy_columns = [
        ConstColumn(
            value=f"CASE WHEN {varname} = '{enum['code']}' THEN 1 ELSE 0 END",
            alias=enum["dummy"],
        )
        for varname in enums.keys()
        for enum in enums[varname]
    ]
    columns = rowid_column + intercept_column + dummy_columns + numerical_columns
    table = Table(name=table_name, columns=columns)
    return table


def get_dummy_encoded_design_matrix_select_stmt(design_matrix_table):
    sel = Select(columns=design_matrix_table.columns, tables=[design_matrix_table])
    return sel.compile()


def get_dummy_encoded_design_matrix_schema(design_matrix_table):
    assert design_matrix_table.columns[0].name == "row_id"
    schema = [("row_id", int)]
    for column in design_matrix_table.columns[1:]:
        colname = column.alias or column.name
        schema.append((colname, float))
    return schema

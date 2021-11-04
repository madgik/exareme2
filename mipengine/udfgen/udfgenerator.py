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

The are two main kinds of UDF input/output type, tensors and relations.  Both
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
generate_udf_queries    Generates a pair of strings holding the UDF definition
                        (when needed) and the query for calling the UDF
TensorUnaryOp           Enum with tensor unary operations
TensorBinaryOp          Enum with tensor binary operations
make_unique_func_name   Helper for creating unique function names
======================= ========================================================
"""
from abc import ABC, abstractmethod
import ast
from copy import deepcopy
from enum import Enum
import inspect
from numbers import Number
from string import Template
import hashlib
import base64

import re
from textwrap import dedent, indent
from typing import (
    Dict,
    List,
    NamedTuple,
    TypeVar,
    Union,
)

import numpy
import astor
from pydantic import BaseModel
from typing import Tuple

from typing import List

from typing import Set

from mipengine.node_tasks_DTOs import ColumnInfo
from mipengine.node_tasks_DTOs import TableInfo
from mipengine import DType as dt

__all__ = [
    "udf",
    "tensor",
    "relation",
    "merge_tensor",
    "scalar",
    "literal",
    "generate_udf_queries",
    "TensorUnaryOp",
    "TensorBinaryOp",
    "make_unique_func_name",
]

# TODO Do not select with star, select columns excplicitly to avoid surprises
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


# TODO refactor these, polymorphism?
def get_build_template(iotype):
    COLUMNS_COMPREHENSION_TMPL = "{{n: _columns[n] for n in {colnames}}}"
    if isinstance(iotype, RelationType):
        return "{name} = pd.DataFrame(" + COLUMNS_COMPREHENSION_TMPL + ")"
    if isinstance(iotype, TensorType):
        return "{name} = udfio.from_tensor_table(" + COLUMNS_COMPREHENSION_TMPL + ")"
    if isinstance(iotype, MergeTensorType):
        return "{name} = udfio.merge_tensor_to_list(" + COLUMNS_COMPREHENSION_TMPL + ")"
    raise NotImplementedError("Build templates only for RelationType and TensorType")


def get_return_stmt_template(iotype):
    if isinstance(iotype, RelationType):
        return "return udfio.as_relational_table(numpy.array({return_name}))"
    if isinstance(iotype, TensorType):
        return "return udfio.as_tensor_table(numpy.array({return_name}))"
    if isinstance(iotype, ScalarType):
        return "return {return_name}"
    if isinstance(iotype, StateObjectType):
        return "return pickle.dumps({return_name})"
    if isinstance(iotype, TransferObjectType):
        return "return {return_name}.json()"
    raise NotImplementedError(
        "Return stmt template only for RelationType, TensorType, ScalarType, ObjectType"
    )


def get_return_type_template(iotype):
    if isinstance(iotype, TableType):
        return f"TABLE({iotype_to_sql_schema(iotype)})"
    if isinstance(iotype, ScalarType):
        return iotype.dtype.to_sql()
    raise NotImplementedError("Return type template only for TableType, ScalarType")


def iotype_to_sql_schema(iotype, name_prefix=""):
    if isinstance(iotype, ScalarType):
        return f"result {iotype.dtype.to_sql()}"
    column_names = iotype.column_names(name_prefix)
    types = [dtype.to_sql() for _, dtype in iotype.schema]
    sql_params = [f"{name} {dtype}" for name, dtype in zip(column_names, types)]
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


def get_return_name_from_body(statements):
    """Returns name of variable in return statement. Assumes that a return
    statemen exists and is of type ast.Name because the validation is supposed
    to happen before (in validate_func_as_udf)."""
    ret_stmt = next(s for s in statements if isinstance(s, ast.Return))
    return ret_stmt.value.id  # type: ignore


def make_unique_func_name(func) -> str:
    """Creates a unique function name composed of the function name, an
    underscore and the module's name hashed, encoded in base32 and truncated at
    4 chars."""
    module_name = func.__module__
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


class TableType(IOType, ABC):
    @property
    @abstractmethod
    def schema(self):
        raise NotImplementedError

    def column_names(self, prefix=""):
        prefix += "_" if prefix else ""
        return [prefix + name for name, _ in self.schema]


class ParametrizedTableType(TableType, ParametrizedType, ABC):
    pass


class TensorType(ParametrizedTableType):
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


class MergeTensorType(ParametrizedTableType):
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


class RelationType(ParametrizedTableType):
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


def relation(schema):
    return RelationType(schema)


class ObjectType(TableType, ABC):
    schema_: List[Tuple]

    def __init__(self, stored_class):
        self._stored_class = stored_class

    @property
    def stored_class(self):
        return self._stored_class

    @property
    def schema(self):
        return self.schema_

    @classmethod
    def schema_matches(cls, schema_provided: List[ColumnInfo]):
        if len(schema_provided) > 1:
            return False
        column_name, column_type = cls.schema_[0]
        column_name_provided = schema_provided[0].name
        column_type_provided = schema_provided[0].dtype
        if column_name != column_name_provided:
            return False
        if column_type != column_type_provided:
            return False
        return True


class TransferObjectType(ObjectType):
    schema_ = [("jsonified_object", dt.JSON)]

    def __init__(self, stored_class):
        if not issubclass(stored_class, BaseModel):
            raise UDFBadDefinition(
                "The 'stored_class' parameter should contain a subclass of BaseModel."
            )
        super().__init__(stored_class)


def transfer_object(stored_class):
    return TransferObjectType(stored_class)


class StateObjectType(ObjectType):
    schema_ = [("pickled_object", dt.BINARY)]

    def __init__(self, stored_class):
        if not inspect.isclass(stored_class):
            raise UDFBadDefinition(
                "The 'stored_class' parameter should contain a class."
            )
        super().__init__(stored_class)


def state_object(stored_class):
    return StateObjectType(stored_class)


class ScalarType(IOType, ParametrizedType):
    def __init__(self, dtype):
        self.dtype = dt.from_py(dtype) if isinstance(dtype, type) else dtype


def scalar(dtype):
    return ScalarType(dtype)


class LiteralType(IOType):
    def __init__(self, value=None) -> None:
        self.value = value


def literal(value=None):
    return LiteralType(value)


# ~~~~~~~~~~~~~~~~~~~~~~~ UDF Arguments ~~~~~~~~~~~~~~~~~~~~~~ #


class UDFArgument:
    __repr__ = recursive_repr
    type: IOType


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


class MergeTensorArg(TableArg):
    def __init__(self, table_name, dtype, ndims):
        self.type = MergeTensorType(dtype, ndims)
        super().__init__(table_name)

    @property
    def ndims(self):
        return self.type.ndims


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


# Oprhan Object Args are ObjectArgs that we don't yet know their class
class OrphanObjectArg(TableArg):
    def __init__(self, table_name):
        super().__init__(table_name)

    @abstractmethod
    def convert_to_object_arg(self, stored_class):
        pass

    def __eq__(self, other):
        if self.table_name != other.table_name:
            return False
        return True


class OrphanStateObjectArg(OrphanObjectArg):
    def __init__(self, table_name):
        super().__init__(table_name)

    def convert_to_object_arg(self, stored_class):
        return StateObjectArg(self, stored_class)


class OrphanTransferObjectArg(OrphanObjectArg):
    def __init__(self, table_name):
        super().__init__(table_name)

    def convert_to_object_arg(self, stored_class):
        return TransferObjectArg(self, stored_class)


class ObjectArg(TableArg, ABC):
    type: ObjectType

    def __init__(self, orphan_object: OrphanObjectArg):
        super().__init__(orphan_object.table_name)

    @property
    def schema(self):
        return self.type.schema

    @property
    def stored_class(self):
        return self.type.stored_class

    def __eq__(self, other):
        if self.table_name != other.table_name:
            return False
        if self.schema != other.schema:
            return False
        if self.type.stored_class != other.type.stored_class:
            return False
        return True


class StateObjectArg(ObjectArg):
    def __init__(self, orphan_object: OrphanStateObjectArg, stored_class):
        self.type = state_object(stored_class)
        super().__init__(orphan_object)


class TransferObjectArg(ObjectArg):
    def __init__(self, orphan_object: OrphanTransferObjectArg, stored_class):
        self.type = transfer_object(stored_class)
        super().__init__(orphan_object)


class LiteralArg(UDFArgument):
    def __init__(self, value):
        self.type: LiteralType = literal(value)

    @property
    def value(self):
        return self.type.value

    def __eq__(self, other):
        return self.value == other.value


# ~~~~~~~~~~~~~~~~~~~~~~~ Type Aliases ~~~~~~~~~~~~~~~~~~~~~~~ #


KnownTypeParams = Union[type, int]
UnknownTypeParams = TypeVar
TypeParamsInference = Dict[UnknownTypeParams, KnownTypeParams]
InputType = Union[TableType, LiteralType]
OutputType = Union[TableType, ScalarType, ObjectType]

# ~~~~~~~~~~~~~~~~~~~~~~ UDF AST Nodes ~~~~~~~~~~~~~~~~~~~~~~~ #


class ASTNode(ABC):
    @abstractmethod
    def compile(self) -> str:
        raise NotImplementedError

    __repr__ = recursive_repr

    def __str__(self):
        return self.compile()  # pragma: no cover


class UDFParameter(ASTNode):
    def __init__(self, param_type, name="") -> None:
        self.name = name
        self.iotype = param_type

    def compile(self) -> str:
        return iotype_to_sql_schema(iotype=self.iotype, name_prefix=self.name)


class UDFReturnType(ASTNode):
    def __init__(self, return_type) -> None:
        self.iotype = return_type

    def compile(self) -> str:
        return get_return_type_template(self.iotype)


class UDFSignature(ASTNode):
    def __init__(
        self,
        udfname: str,  # unused as long as generator returns templates
        parameter_types: Dict[str, ParametrizedTableType],
        return_type: Union[TableType, ScalarType],
    ):
        self.udfname = "$udf_name"
        self.parameter_types = [
            UDFParameter(type_, name) for name, type_ in parameter_types.items()
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
    def __init__(self, udfname, param_table_types, return_type):
        self.signature = UDFSignature(udfname, param_table_types, return_type)

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
    def __init__(self, pickle, pydantic):
        self._import_lines = ["import pandas as pd", "import udfio"]
        if pickle:
            pickle = "import dill as pickle"
            self._import_lines.append(pickle)
        if pydantic:
            pydantic = "from pydantic import BaseModel"
            self._import_lines.append(pydantic)
            typing = "from typing import List, Dict"
            self._import_lines.append(typing)

    def compile(self) -> str:
        return LN.join(self._import_lines)


class ClassDefinitions(ASTNode):
    def __init__(self, parameter_types: Dict[str, IOType], return_type: IOType):
        self._class_definitions: Set[str] = set()
        for io_type in list(parameter_types.values()) + [return_type]:
            if isinstance(io_type, ObjectType):
                class_def = dedent(inspect.getsource(io_type.stored_class))
                self._class_definitions.add(class_def)

    def compile(self) -> str:
        class_defs = LN.join(self._class_definitions)
        return class_defs if class_defs else ""


class TableBuild(ASTNode):
    def __init__(self, table_name, table, template):
        self.table_name = table_name
        self.table = table
        self.template = template

    def compile(self) -> str:
        colnames = self.table.column_names(prefix=self.table_name)
        return self.template.format(name=self.table_name, colnames=colnames)


class TableBuilds(ASTNode):
    def __init__(self, parameter_types):
        self.table_builds = [
            TableBuild(name, param_type, template=get_build_template(param_type))
            for name, param_type in parameter_types.items()
        ]

    def compile(self) -> str:
        return LN.join([tb.compile() for tb in self.table_builds])


class UDFReturnStatement(ASTNode):
    def __init__(self, return_name, return_type):
        self.return_name = return_name
        self.template = get_return_stmt_template(return_type)

    def compile(self) -> str:
        return self.template.format(return_name=self.return_name)


class LiteralAssignments(ASTNode):
    def __init__(self, literals):
        self.literals = literals

    def compile(self) -> str:
        return LN.join(f"{name} = {arg.value}" for name, arg in self.literals.items())


class ObjectAssignments(ASTNode):
    def __init__(self, objects: List[ObjectType]):
        self.object_assignments = []
        for object_name, object_type in objects.items():
            loopback_query = (
                f"state_str = "
                f"_conn.execute(\"SELECT jsonified_object from TODOOOO;\")['jsonified_object'][0]"
            )
            object_parse = f"{object_name} = {object_type.stored_class.__name__}.parse_raw(state_str)"
            self.object_assignments.append(LN.join([loopback_query, object_parse]))

    def compile(self) -> str:
        return LN.join(self.object_assignments)


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
        param_table_types: Dict[str, IOType],
        statements: list,
        return_name: str,
        return_type: IOType,
        literals,
        objects: Dict[str, IOType],
    ):
        self.returnless_stmts = UDFBodyStatements(statements)
        self.return_stmt = UDFReturnStatement(return_name, return_type)
        self.table_conversions = TableBuilds(param_table_types)
        self.literals = LiteralAssignments(literals)
        self.imports = Imports(
            UDFBody._is_pickle_needed(param_table_types, return_type),
            UDFBody._is_pydantic_needed(param_table_types, return_type),
        )
        self.class_definitions = ClassDefinitions(param_table_types, return_type)
        self.objects = ObjectAssignments(objects)

    @classmethod
    def _is_pickle_needed(cls, parameter_types: Dict[str, IOType], return_type: IOType):
        for io_type in list(parameter_types.values()) + [return_type]:
            if isinstance(io_type, StateObjectType):
                return True
        return False

    @classmethod
    def _is_pydantic_needed(
        cls, parameter_types: Dict[str, IOType], return_type: IOType
    ):
        for io_type in list(parameter_types.values()) + [return_type]:
            if isinstance(io_type, TransferObjectType):
                return True
        return False

    def compile(self) -> str:
        return LN.join(
            remove_empty_lines(
                [
                    self.imports.compile(),
                    self.class_definitions.compile(),
                    self.table_conversions.compile(),
                    self.literals.compile(),
                    self.objects.compile(),
                    self.returnless_stmts.compile(),
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
        param_table_types: Dict[str, ParametrizedTableType],
        output_type: Union[TableType, ScalarType],
        literal_types: Dict[str, LiteralType],
        object_types: Dict[str, ObjectType],
        traceback=False,
    ):
        self.header = UDFHeader(
            udfname=funcparts.qualname,
            param_table_types=param_table_types,
            return_type=output_type,
        )
        body = UDFBody(
            param_table_types=param_table_types,
            statements=funcparts.body_statements,
            return_name=funcparts.return_name,
            return_type=funcparts.output_type,
            literals=literal_types,
            objects=object_types,
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
    def __init__(self, name: str, columns: List[Column], alias=""):
        self.name = name
        self.columns = {colname: Column(colname, self) for colname in columns}
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


class FromClause(ASTNode):
    def __init__(self, tables, use_alias=True):
        self.tables = tables
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
            if not decorator_parameter_names_are_valid(parameter_names, kwargs):
                raise UDFBadDefinition(
                    f"Invalid parameter names in udf decorator of {func}: "
                    f"parameter names: {parameter_names}, "
                    f"decorator kwargs: {kwargs}."
                )
            signature = make_udf_signature(parameter_names, kwargs)
            validate_udf_signature_types(signature)
            validate_udf_return_statement(func)
            funcparts = breakup_function(func, signature)
            validate_udf_table_input_types(funcparts.param_table_input_types)
            funcname = funcparts.qualname
            if funcname in self.registry:
                raise UDFBadDefinition(
                    f"A function named {funcname} is already in the udf registry."
                )
            self.registry[funcname] = funcparts
            return func

        return decorator


# Singleton pattern
udf = UDFDecorator()
del UDFDecorator


class Signature(NamedTuple):
    parameters: Dict[str, IOType]
    return_annotation: IOType


class FunctionParts(NamedTuple):
    """A function's parts, used in various stages of the udf definition/query
    generation."""

    qualname: str
    body_statements: list
    return_name: str
    param_table_input_types: Dict[str, ParametrizedTableType]
    object_input_types: Dict[str, ObjectType]
    literal_input_types: Dict[str, LiteralType]
    output_type: IOType
    sig: Signature


class UDFBadDefinition(Exception):
    """Raised when an error is detected in the definition of a udf decorated
    function. These checks are made as soon as the function is defined."""


def decorator_parameter_names_are_valid(parameter_names, decorator_kwargs):
    """Returns False if parameter_names are not contained in decorator_kwargs
    or if 'return_type' is not contained in decorator_kwargs, True
    otherwise."""
    if not set(parameter_names) <= set(decorator_kwargs):
        return False
    if "return_type" not in decorator_kwargs:
        return False
    return True


def make_udf_signature(parameter_names, decorator_kwargs):
    parameters = {name: decorator_kwargs[name] for name in parameter_names}
    return_annotation = decorator_kwargs["return_type"]
    signature = Signature(parameters, return_annotation)
    return signature


def validate_udf_signature_types(funcsig):
    """Validates that all types used in the udf's type signature, both input
    and output, are subclasses of IOType."""
    parameter_types = funcsig.parameters.values()
    return_type = funcsig.return_annotation
    if any(not isinstance(input_type, IOType) for input_type in parameter_types):
        raise UDFBadDefinition(
            f"Input types of func are not subclasses of IOType: {parameter_types}."
        )
    if not isinstance(return_type, IOType):
        raise UDFBadDefinition(
            f"Output type of func is not subclass of IOType: {return_type}."
        )


def validate_udf_return_statement(func):
    """Validates two things concerning the return statement of a udf. 1) that
    there is one and 2) that it is of the simple `return name` form, as no
    expressions are allowd in udf return statements."""
    tree = parse_func(func)
    statements = get_func_body_from_ast(tree)
    try:
        ret_stmt = next(s for s in statements if isinstance(s, ast.Return))
    except StopIteration as stop_iter:
        raise UDFBadDefinition(f"Return statement not found in {func}.") from stop_iter
    if not isinstance(ret_stmt.value, ast.Name):
        raise UDFBadDefinition(
            f"Expression in return statement in {func}."
            "Assign expression to variable and return it."
        )


def breakup_function(func, funcsig) -> FunctionParts:
    """Breaks up a function into smaller parts, which will be used during
    the udf translation process."""
    qualname = make_unique_func_name(func)
    tree = parse_func(func)
    body_statements = get_func_body_from_ast(tree)
    return_name = get_return_name_from_body(body_statements)
    param_table_input_types = {
        name: iotype
        for name, iotype in funcsig.parameters.items()
        if isinstance(iotype, ParametrizedTableType)
    }
    object_input_types = {
        name: iotype
        for name, iotype in funcsig.parameters.items()
        if isinstance(iotype, ObjectType)
    }
    literal_input_types = {
        name: iotype
        for name, iotype in funcsig.parameters.items()
        if isinstance(iotype, LiteralType)
    }
    output_type = funcsig.return_annotation
    return FunctionParts(
        qualname,
        body_statements,
        return_name,
        param_table_input_types,
        object_input_types,
        literal_input_types,
        output_type,
        funcsig,
    )


def validate_udf_table_input_types(table_input_types):
    tensors = get_items_of_type(TensorType, table_input_types)
    relations = get_items_of_type(RelationType, table_input_types)
    if tensors and relations:
        raise UDFBadDefinition("Cannot pass both tensors and relations to udf.")


# ~~~~~~~~~~~~~~~~~ Module Public Function ~~~~~~~~~~~~~~~ #


LiteralValue = Union[Number, numpy.ndarray]
UDFGenArgument = Union[TableInfo, LiteralValue]


class UDFBadCall(Exception):
    """Raised when something is wrong with the arguments passed to the udf
    generator."""


def generate_udf_queries(
    func_name: str,
    positional_args: List[UDFGenArgument],
    keyword_args: Dict[str, UDFGenArgument],
    traceback=False,
):
    udf_posargs, udf_kwargs = convert_udfgenargs_to_udfargs(
        positional_args,
        keyword_args,
    )
    # --> Hack for merge tensors since we don't tag TableInfos with different kinds
    if "sum_tensors" in func_name:
        tensor_arg, *_ = udf_posargs
        table_name = tensor_arg.table_name
        dtype = tensor_arg.dtype
        ndims = tensor_arg.ndims
        udf_posargs = [MergeTensorArg(table_name, dtype=dtype, ndims=ndims)]
    # <--

    if func_name in TENSOR_OP_NAMES:
        udf_definition = ""  # TODO this looks stupid, Node should know what to expect
        udf_select = get_sql_tensor_operation_select_query(udf_posargs, func_name)
        output_type = get_output_type_for_sql_tensor_operation(func_name, udf_posargs)
        udf_execution_query = get_udf_create_and_insert_template(
            output_type, udf_select
        )
        return Template(udf_definition), Template(udf_execution_query)

    udf_definition, udf_execution_query = get_udf_templates_using_udfregistry(
        funcname=func_name,
        posargs=udf_posargs,
        keywordargs=udf_kwargs,
        udfregistry=udf.registry,
        traceback=traceback,
    )
    return Template(udf_definition), Template(udf_execution_query)


def convert_udfgenargs_to_udfargs(udfgen_posargs, udfgen_kwargs):
    udf_posargs = [convert_udfgenarg_to_udfarg(arg) for arg in udfgen_posargs]
    udf_keywordargs = {
        name: convert_udfgenarg_to_udfarg(arg) for name, arg in udfgen_kwargs
    }
    return udf_posargs, udf_keywordargs


def convert_udfgenarg_to_udfarg(udfgen_arg) -> UDFArgument:
    if isinstance(udfgen_arg, TableInfo):
        return convert_table_info_to_table_arg(udfgen_arg)
    return LiteralArg(value=udfgen_arg)


def convert_table_info_to_table_arg(table_info):
    "add new input args"
    if TransferObjectType.schema_matches(table_info.schema_.columns):
        return OrphanTransferObjectArg(table_name=table_info.name)
    elif StateObjectType.schema_matches(table_info.schema_.columns):
        return OrphanStateObjectArg(table_name=table_info.name)
    elif is_tensor_schema(table_info.schema_.columns):
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

# TODO is_***_schema should probably be moved in the type class as class methods
def is_tensor_schema(schema):
    colnames = [col.name for col in schema]
    if "val" in colnames and any(cname.startswith("dim") for cname in colnames):
        return True
    return False


def convert_table_schema_to_relation_schema(table_schema):
    return [(c.name, c.dtype) for c in table_schema if c.name != ROWID]


# <--


def get_udf_templates_using_udfregistry(
    funcname: str,
    posargs: List[UDFArgument],
    keywordargs: Dict[str, UDFArgument],
    udfregistry: dict,
    traceback=False,
) -> tuple:
    funcparts = get_funcparts_from_udf_registry(funcname, udfregistry)
    udf_args = get_udf_args(funcparts, posargs, keywordargs)
    udf_args = assign_class_to_orphan_object_args(
        udf_args, funcparts.object_input_types
    )
    input_types = copy_types_from_udfargs(udf_args)
    output_type = get_output_type(funcparts, input_types, traceback)
    udf_definition = get_udf_definition_template(
        funcparts,
        input_types,
        output_type,
        traceback=traceback,
    )
    table_args = get_items_of_type(TableArg, mapping=udf_args)
    udf_select = get_udf_select_template(
        ScalarType(str) if traceback else output_type,
        table_args,
    )
    udf_execution_query = get_udf_create_and_insert_template(output_type, udf_select)
    return udf_definition, udf_execution_query


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


def get_udf_args(funcparts, posargs, keywordargs):
    udf_args = merge_args_and_kwargs(
        param_names=funcparts.sig.parameters.keys(),
        args=posargs,
        kwargs=keywordargs,
    )
    validate_arg_names(udf_args, funcparts.sig.parameters)
    return udf_args


def assign_class_to_orphan_object_args(
    udf_args_provided: Dict[str, UDFArgument], object_input_types: Dict[str, ObjectType]
):
    udf_args = {}
    for param_name, arg in udf_args_provided.items():
        if isinstance(arg, OrphanObjectArg):
            stored_class = object_input_types[param_name].stored_class
            udf_args[param_name] = arg.convert_to_object_arg(stored_class)
        else:
            udf_args[param_name] = arg
    return udf_args


def get_funcparts_from_udf_registry(funcname: str, udfregistry: dict) -> FunctionParts:
    if funcname not in udfregistry:
        raise UDFBadCall(f"{funcname} cannot be found in udf registry.")
    return udfregistry[funcname]


def validate_arg_names(
    args: Dict[str, UDFArgument],
    parameters: Dict[str, IOType],
) -> None:
    """Validates that the names of the udf arguments are the expected ones,
    based on the udf's formal parameters."""
    if args.keys() != parameters.keys():
        raise UDFBadCall(
            f"UDF argument names do not match UDF parameter names: "
            f"{args.keys()}, {parameters.keys()}."
        )


def copy_types_from_udfargs(udfargs: Dict[str, UDFArgument]) -> Dict[str, IOType]:
    return {name: deepcopy(arg.type) for name, arg in udfargs.items()}


# ~~~~~~~~~~~~~~~ UDF Definition Translator ~~~~~~~~~~~~~~ #


def get_udf_definition_template(
    funcparts: FunctionParts,
    input_types: Dict[str, IOType],
    output_type,
    traceback=False,
) -> str:
    param_table_types = get_items_of_type(ParametrizedTableType, mapping=input_types)
    object_types = get_items_of_type(ObjectType, mapping=input_types)
    literal_types = get_items_of_type(LiteralType, mapping=input_types)
    verify_declared_and_passed_types_match(
        funcparts.param_table_input_types, param_table_types
    )
    udf_definition = UDFDefinition(
        funcparts=funcparts,
        param_table_types=param_table_types,
        output_type=output_type,
        literal_types=literal_types,
        object_types=object_types,
        traceback=traceback,
    )
    return udf_definition.compile()


def get_output_type(funcparts, input_types, traceback) -> OutputType:
    """Computes the UDF output type. If `traceback` is true the type is str
    since the traceback will be returned as a string. If the output type is
    generic its type parameters must be inferred from the passed input types.
    Otherwise, the declared output type is returned."""
    if traceback:
        return scalar(str)
    if (
        isinstance(funcparts.output_type, ParametrizedType)
        and funcparts.output_type.is_generic
    ):
        param_table_types = get_items_of_type(
            ParametrizedTableType, mapping=input_types
        )
        return infer_output_type(
            passed_input_types=param_table_types,
            declared_input_types=funcparts.param_table_input_types,
            declared_output_type=funcparts.output_type,
        )
    return funcparts.output_type


def verify_declared_and_passed_types_match(
    declared_types: Dict[str, ParametrizedTableType],
    passed_types: Dict[str, ParametrizedTableType],
) -> None:
    for paramname, param in passed_types.items():
        known_params = declared_types[paramname].known_typeparams
        verify_declared_typeparams_match_passed_type(known_params, param)


def verify_declared_typeparams_match_passed_type(
    known_typeparams: Dict[str, KnownTypeParams],
    passed_type: IOType,
) -> None:
    for name, param in known_typeparams.items():
        if not hasattr(passed_type, name):
            raise UDFBadCall(f"{passed_type} has no typeparam {name}.")
        if getattr(passed_type, name) != param:
            raise UDFBadCall(
                "IOType's known typeparams do not match typeparams passed "
                f"in {passed_type}: {param}, {getattr(passed_type, name)}."
            )


def infer_output_type(
    passed_input_types: Dict[str, ParametrizedTableType],
    declared_input_types: Dict[str, ParametrizedTableType],
    declared_output_type: ParametrizedType,
) -> Union[ParametrizedTableType, ScalarType]:
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
    declared_input_types: Dict[str, ParametrizedTableType],
    passed_input_types: Dict[str, ParametrizedTableType],
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


def get_udf_select_template(output_type: IOType, table_args: Dict[str, TableArg]):
    tensors = get_table_ast_nodes_from_table_args(
        table_args,
        arg_type=(TensorArg, MergeTensorArg),
    )
    relations = get_table_ast_nodes_from_table_args(table_args, arg_type=RelationArg)
    tables = tensors or relations
    columns = [column for table in tables for column in table.columns.values()]
    where_clause = get_where_clause_for_tensors(tensors) if tensors else None
    if isinstance(output_type, ScalarType):
        func = ScalarFunction(name="$udf_name", columns=columns)
        select_stmt = Select([func], tables, where_clause)
        return select_stmt.compile()
    if isinstance(output_type, TableType):
        subquery = Select(columns, tables, where_clause) if tables else None
        func = TableFunction(name="$udf_name", subquery=subquery)
        select_stmt = Select([nodeid_column(), Column("*")], [func])
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


def convert_table_arg_to_table_ast_node(table_arg, alias=None):
    return Table(
        name=table_arg.table_name,
        columns=table_arg.column_names(),
        alias=alias,
    )


def nodeid_column():
    return ScalarFunction(
        name="CAST",
        columns=[Column("'$node_id'", alias=dt.STR.to_sql())],
        alias="node_id",
    )


# ~~~~~~~~~~~~~~~~~ CREATE TABLE query generator ~~~~~~~~~ #


def get_udf_create_and_insert_template(output_type, udf_select_query):
    table_name = "$table_name"
    script = [DROP_TABLE_IF_EXISTS + " " + table_name + SCOLON]
    output_schema = iotype_to_sql_schema(output_type)
    if not isinstance(output_type, ScalarType):
        output_schema = f"node_id {dt.STR.to_sql()}," + output_schema
    script += [CREATE_TABLE + " " + table_name + f"({output_schema})" + SCOLON]
    script += [f"INSERT INTO {table_name}\n" + udf_select_query + SCOLON]
    return LN.join(script)


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

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

>>> @udf(x=tensor(dtype=int, ndims=1), return_type=relation([("sum", int)]))
... def sum_vector(x):
...     result = sum(x)
...     return result

Let's look at the decorator

@udf(x=tensor(dtype=int, ndims=1), return_type=relation([("sum", int)]))
     ^                             ^
     |                             |
     The function takes an         The function returns a
     argument x of type            "table with one column
     "one dimensional tensor       of datatype int"
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
literal                 Literal type factory
state                   State type factory
transfer                Transfer type factory
merge_transfer          Merge transfer type factory
generate_udf_queries    Generates a pair of strings holding the UDF definition
                        (when needed) and the query for calling the UDF
make_unique_func_name   Helper for creating unique function names
======================= ========================================================
"""
from copy import deepcopy
from numbers import Number
from string import Template
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union

from mipengine.node_tasks_DTOs import SMPCTablesInfo
from mipengine.node_tasks_DTOs import TableInfo
from mipengine.node_tasks_DTOs import TableType as DBTableType
from mipengine.udfgen.ast import Column
from mipengine.udfgen.ast import ColumnEqualityClause
from mipengine.udfgen.ast import ConstColumn
from mipengine.udfgen.ast import FunctionParts
from mipengine.udfgen.ast import Select
from mipengine.udfgen.ast import StarColumn
from mipengine.udfgen.ast import Table
from mipengine.udfgen.ast import TableFunction
from mipengine.udfgen.ast import UDFDefinition
from mipengine.udfgen.ast import _get_loopback_tables_template_names
from mipengine.udfgen.decorator import udf
from mipengine.udfgen.helpers import compose_mappings
from mipengine.udfgen.helpers import get_items_of_type
from mipengine.udfgen.helpers import iotype_to_sql_schema
from mipengine.udfgen.helpers import mapping_inverse
from mipengine.udfgen.helpers import merge_args_and_kwargs
from mipengine.udfgen.helpers import merge_mappings_consistently
from mipengine.udfgen.iotypes import MAIN_TABLE_PLACEHOLDER
from mipengine.udfgen.iotypes import InputType
from mipengine.udfgen.iotypes import LiteralArg
from mipengine.udfgen.iotypes import LiteralType
from mipengine.udfgen.iotypes import LoopbackOutputType
from mipengine.udfgen.iotypes import MergeTensorType
from mipengine.udfgen.iotypes import MergeTransferType
from mipengine.udfgen.iotypes import OutputType
from mipengine.udfgen.iotypes import ParametrizedType
from mipengine.udfgen.iotypes import PlaceholderArg
from mipengine.udfgen.iotypes import PlaceholderType
from mipengine.udfgen.iotypes import RelationArg
from mipengine.udfgen.iotypes import SecureTransferArg
from mipengine.udfgen.iotypes import SecureTransferType
from mipengine.udfgen.iotypes import SMPCSecureTransferArg
from mipengine.udfgen.iotypes import StateArg
from mipengine.udfgen.iotypes import StateType
from mipengine.udfgen.iotypes import TableArg
from mipengine.udfgen.iotypes import TableType
from mipengine.udfgen.iotypes import TensorArg
from mipengine.udfgen.iotypes import TransferArg
from mipengine.udfgen.iotypes import TransferType
from mipengine.udfgen.iotypes import UDFArgument
from mipengine.udfgen.iotypes import UDFLoggerArg
from mipengine.udfgen.iotypes import _get_smpc_table_template_names
from mipengine.udfgen.iotypes import merge_tensor
from mipengine.udfgen.iotypes import merge_transfer
from mipengine.udfgen.iotypes import relation
from mipengine.udfgen.udfgen_DTOs import UDFGenExecutionQueries
from mipengine.udfgen.udfgen_DTOs import UDFGenResult
from mipengine.udfgen.udfgen_DTOs import UDFGenSMPCResult
from mipengine.udfgen.udfgen_DTOs import UDFGenTableResult

KnownTypeParams = Union[type, int]
UnknownTypeParams = TypeVar
TypeParamsInference = Dict[UnknownTypeParams, KnownTypeParams]
LiteralValue = Union[Number, str, list, dict]
UDFGenArgument = Union[TableInfo, LiteralValue, SMPCTablesInfo]


class UDFBadCall(Exception):
    """Raised when something is wrong with the arguments passed to the udf
    generator."""


def generate_udf_queries(
    func_name: str,
    positional_args: List[UDFGenArgument],
    keyword_args: Dict[str, UDFGenArgument],
    smpc_used: bool,
    output_schema=None,
) -> UDFGenExecutionQueries:
    """
    Parameters
    ----------
    func_name: The name of the udf to run
    positional_args: Positional arguments
    keyword_args: Keyword arguments
    smpc_used: Is SMPC used in the computations?

    Returns
    -------
    a UDFExecutionQueries object.

    """
    udf_posargs, udf_kwargs = convert_udfgenargs_to_udfargs(
        positional_args,
        keyword_args,
        smpc_used,
    )

    if func_name == "create_dummy_encoded_design_matrix":
        return get_create_dummy_encoded_design_matrix_execution_queries(keyword_args)

    return get_udf_templates_using_udfregistry(
        funcname=func_name,
        posargs=udf_posargs,
        keywordargs=udf_kwargs,
        udfregistry=udf.registry,
        smpc_used=smpc_used,
        output_schema=output_schema,
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


def convert_udfgenarg_to_udfarg(udfgen_arg: UDFGenArgument, smpc_used) -> UDFArgument:
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
    if smpc_udf_input.sum_op:
        sum_op_table_name = smpc_udf_input.sum_op.name
    if smpc_udf_input.min_op:
        min_op_table_name = smpc_udf_input.min_op.name
    if smpc_udf_input.max_op:
        max_op_table_name = smpc_udf_input.max_op.name
    return SMPCSecureTransferArg(
        template_table_name=smpc_udf_input.template.name,
        sum_op_values_table_name=sum_op_table_name,
        min_op_values_table_name=min_op_table_name,
        max_op_values_table_name=max_op_table_name,
    )


def convert_table_info_to_table_arg(table_info: TableInfo, smpc_used):
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
        return get_tensor_arg_from_table_info(table_info)
    relation_schema = convert_table_schema_to_relation_schema(
        table_info.schema_.columns
    )
    return RelationArg(table_name=table_info.name, schema=relation_schema)


def get_tensor_arg_from_table_info(table_info):
    ndims = sum(1 for col in table_info.schema_.columns if col.name.startswith("dim"))
    valcol = next(col for col in table_info.schema_.columns if col.name == "val")
    dtype = valcol.dtype
    return TensorArg(table_name=table_info.name, dtype=dtype, ndims=ndims)


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
    return [(c.name, c.dtype) for c in table_schema]


def get_udf_templates_using_udfregistry(
    funcname: str,
    posargs: List[UDFArgument],
    keywordargs: Dict[str, UDFArgument],
    udfregistry: dict,
    smpc_used: bool,
    output_schema=None,
) -> UDFGenExecutionQueries:
    funcparts = get_funcparts_from_udf_registry(funcname, udfregistry)
    udf_args = get_udf_args(funcparts, posargs, keywordargs)
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
        funcparts,
        udf_args,
        output_schema,
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
    )

    table_args = get_items_of_type(TableArg, mapping=udf_args)
    udf_select = get_udf_select_template(
        main_output_type,
        table_args,
    )
    udf_execution = get_udf_execution_template(udf_select)

    return UDFGenExecutionQueries(
        udf_results=udf_outputs,
        udf_definition_query=udf_definition,
        udf_select_query=udf_execution,
    )


def get_udf_args(funcparts, posargs, keywordargs) -> Dict[str, UDFArgument]:
    udf_args = merge_args_and_kwargs(
        param_names=funcparts.sig.parameters.keys(),
        args=posargs,
        kwargs=keywordargs,
    )

    # Check logger_param_name argument is not given and if not, create it.
    if funcparts.logger_param_name:
        if funcparts.logger_param_name in udf_args.keys():
            raise UDFBadCall(
                "No argument should be provided for "
                f"'UDFLoggerType' parameter: '{funcparts.logger_param_name}'"
            )
        udf_args[funcparts.logger_param_name] = UDFLoggerArg(udf_name="$udf_name")
    placeholders = get_items_of_type(PlaceholderType, funcparts.sig.parameters)
    if placeholders:
        udf_args.update(
            {
                name: PlaceholderArg(type=placeholder)
                for name, placeholder in placeholders.items()
            }
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
    placeholder_args = get_items_of_type(PlaceholderArg, input_args)

    verify_declared_and_passed_param_types_match(
        funcparts.table_input_types, table_args
    )
    udf_definition = UDFDefinition(
        udfname="$udf_name",
        funcparts=funcparts,
        table_args=table_args,
        smpc_args=smpc_args,
        literal_args=literal_args,
        logger_arg=logger_arg,
        placeholder_args=placeholder_args,
        main_output_type=main_output_type,
        sec_output_types=sec_output_types,
        smpc_used=smpc_used,
    )
    return Template(udf_definition.compile())


def get_output_types(
    funcparts,
    udf_args,
    output_schema,
) -> Tuple[OutputType, List[LoopbackOutputType]]:
    """Computes  the  UDF  output  type. If the output type is generic its
    type  parameters  must  be  inferred  from  the  passed  input  types.
    Otherwise, the declared output type is returned."""
    input_types = copy_types_from_udfargs(udf_args)

    if output_schema:
        main_output_type = relation(schema=output_schema)
        return main_output_type, []
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
) -> ParametrizedType:
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
    tensors = get_table_ast_nodes_from_table_args(table_args, arg_type=TensorArg)
    relations = get_table_ast_nodes_from_table_args(table_args, arg_type=RelationArg)
    tables = tensors or relations
    columns = [column for table in tables for column in table.columns.values()]
    where_clause = get_where_clause_for_tensors(tensors) if tensors else None
    where_clause = (
        get_where_clause_for_relations(relations) if relations else where_clause
    )
    subquery = Select(columns, tables, where_clause) if tables else None
    func = TableFunction(name="$udf_name", subquery=subquery)
    select_stmt = Select([StarColumn()], [func])
    return select_stmt.compile()


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


# ~~~~~~~~~~~~~~~~~ CREATE TABLE and INSERT query generator ~~~~~~~~~ #
def _create_table_udf_output(
    output_type: OutputType,
    table_name: str,
) -> UDFGenResult:
    drop_table = "DROP TABLE IF EXISTS" + " $" + table_name + ";"
    output_schema = iotype_to_sql_schema(output_type)
    create_table = "CREATE TABLE" + " $" + table_name + f"({output_schema})" + ";"
    return UDFGenTableResult(
        tablename_placeholder=table_name,
        table_schema=output_type.schema,
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
    return UDFGenSMPCResult(
        template=template,
        sum_op_values=sum_op,
        min_op_values=min_op,
        max_op_values=max_op,
    )


def _create_udf_output(
    output_type: OutputType,
    table_name: str,
    smpc_used: bool,
) -> UDFGenResult:
    if isinstance(output_type, SecureTransferType) and smpc_used:
        return _create_smpc_udf_output(output_type, table_name)
    else:
        return _create_table_udf_output(output_type, table_name)


def get_udf_outputs(
    main_output_type: OutputType,
    sec_output_types: List[LoopbackOutputType],
    smpc_used: bool,
) -> List[UDFGenResult]:
    table_name = MAIN_TABLE_PLACEHOLDER
    udf_outputs = [_create_udf_output(main_output_type, table_name, smpc_used)]

    if sec_output_types:
        for table_name, sec_output_type in _get_loopback_tables_template_names(
            sec_output_types
        ):
            udf_outputs.append(
                _create_udf_output(sec_output_type, table_name, smpc_used)
            )

    return udf_outputs


def get_udf_execution_template(udf_select_query) -> Template:
    table_name = MAIN_TABLE_PLACEHOLDER
    udf_execution = f"INSERT INTO ${table_name}\n" + udf_select_query + ";"
    return Template(udf_execution)


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

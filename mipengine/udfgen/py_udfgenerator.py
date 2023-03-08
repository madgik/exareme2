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

NOTE: Even though it's not mandatory, it's faster to have state as the first
input/output.


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
...     transfer["sum"] = {"data": x["key"] + y["key"], "operation": "sum",
...                        "type": "float"}
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
import functools
from copy import deepcopy
from numbers import Number
from string import Template
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from mipengine.node_tasks_DTOs import SMPCTablesInfo
from mipengine.node_tasks_DTOs import TableInfo
from mipengine.node_tasks_DTOs import TableType as DBTableType
from mipengine.udfgen.ast import CreateTable
from mipengine.udfgen.ast import FunctionParts
from mipengine.udfgen.ast import Insert
from mipengine.udfgen.ast import Select
from mipengine.udfgen.ast import StarColumn
from mipengine.udfgen.ast import Table
from mipengine.udfgen.ast import TableFunction
from mipengine.udfgen.ast import UDFBody
from mipengine.udfgen.ast import UDFDefinition
from mipengine.udfgen.ast import UDFHeader
from mipengine.udfgen.decorator import UDFBadCall
from mipengine.udfgen.decorator import UdfRegistry
from mipengine.udfgen.helpers import get_items_of_type
from mipengine.udfgen.helpers import merge_args_and_kwargs
from mipengine.udfgen.iotypes import InputType
from mipengine.udfgen.iotypes import LiteralArg
from mipengine.udfgen.iotypes import LoopbackOutputType
from mipengine.udfgen.iotypes import MergeTensorType
from mipengine.udfgen.iotypes import MergeTransferType
from mipengine.udfgen.iotypes import OutputType
from mipengine.udfgen.iotypes import ParametrizedType
from mipengine.udfgen.iotypes import PlaceholderArg
from mipengine.udfgen.iotypes import PlaceholderType
from mipengine.udfgen.iotypes import RelationArg
from mipengine.udfgen.iotypes import StateArg
from mipengine.udfgen.iotypes import StateType
from mipengine.udfgen.iotypes import TableArg
from mipengine.udfgen.iotypes import TableType
from mipengine.udfgen.iotypes import TensorArg
from mipengine.udfgen.iotypes import TransferArg
from mipengine.udfgen.iotypes import TransferType
from mipengine.udfgen.iotypes import UDFArgument
from mipengine.udfgen.iotypes import UDFLoggerArg
from mipengine.udfgen.iotypes import merge_tensor
from mipengine.udfgen.iotypes import merge_transfer
from mipengine.udfgen.iotypes import relation
from mipengine.udfgen.smpc import SecureTransferArg
from mipengine.udfgen.smpc import SecureTransferType
from mipengine.udfgen.smpc import SMPCSecureTransferArg
from mipengine.udfgen.smpc import SMPCSecureTransferType
from mipengine.udfgen.smpc import UDFBodySMPC
from mipengine.udfgen.smpc import get_smpc_tablename_placeholders
from mipengine.udfgen.typeinference import infer_output_type
from mipengine.udfgen.udfgen_DTOs import UDFGenResult
from mipengine.udfgen.udfgen_DTOs import UDFGenSMPCResult
from mipengine.udfgen.udfgen_DTOs import UDFGenTableResult

LiteralValue = Union[Number, str, list, dict]
FlowUdfArg = Union[TableInfo, LiteralValue, SMPCTablesInfo]


class UdfGenerator:
    """Generator for MonetDB Python UDFs

    The generator operates starting from a python functions, decorated with the
    `udf` decorator. Generator objects are instantiated based on a particular
    function, found in a passed `UdfRegistry`, and on a set of arguments found
    in the algorithm flow. Some additional flags and parameters are also
    necessary. Then, the generator object has methods for generating two types
    of queries, UDF definitions and UDF execution statements (basically a
    SELECT wrapped in an INSERT statement), as well as objects representing the
    resulting tables.
    """

    def __init__(
        self,
        udfregistry: UdfRegistry,
        func_name: str,
        flowargs: List[FlowUdfArg],
        flowkwargs: Dict[str, FlowUdfArg],
        smpc_used: bool = False,
        request_id: Optional[str] = None,
        output_schema=None,
        min_row_count: int = None,
    ):
        """
        Parameters
        ----------
        udfregistry : UdfRegistry
            Data structure containing python functions registered as UDFs
        func_name : str
            Unique key of a function in `udfregistry`
        flowargs : List[FlowUdfArg]
            Positional arguments received by the algorithm flow
        flowkwargs : Dict[str, FlowUdfArg]
            Keyword arguments received by the algorithm flow
        smpc_used : bool
            True when SMPC framework is used
        request_id : Optional[str]
            Request id
        output_schema : List[Tuple[str, DType]]
            This argument needs to be provided when the output schema is declared as
            deferred
        min_row_count : int
            Minimum allowed number of rows for an input table
        """
        self.func_name = func_name
        self.smpc_used = smpc_used
        self.request_id = request_id
        self.output_schema = output_schema
        self.min_row_count = min_row_count

        self.funcparts = udfregistry[func_name]
        if smpc_used:
            cast_secure_transfers_for_smpc(self.funcparts.output_types)

        self.udf_args = self._get_udf_args(flowargs, flowkwargs)

    @functools.cached_property
    def output_types(self) -> Tuple[OutputType, List[LoopbackOutputType]]:
        """Computes the UDF output type.

        There are three cases:
            - The output type is `RelationType` whose `schema` is DEFERRED
            - The output type is inferred dynamically
            - The output type is known statically
        """

        input_types = copy_types_from_udfargs(self.udf_args)

        # case: output type has DEFERRED schema
        if self.output_schema:
            main_output_type = relation(schema=self.output_schema)
            return (main_output_type,)

        # case: output type has to be infered at execution time
        main_output_type, *_ = self.funcparts.output_types
        if (
            isinstance(main_output_type, ParametrizedType)
            and main_output_type.is_generic
        ):
            param_table_types = get_items_of_type(TableType, mapping=input_types)
            main_output_type = infer_output_type(
                passed_input_types=param_table_types,
                declared_input_types=self.funcparts.table_input_types,
                declared_output_type=main_output_type,
            )
            return (main_output_type,)

        # case: output types are known
        return self.funcparts.output_types

    def get_definition(
        self,
        udf_name: str,
        output_table_names: Optional[List[str]] = None,
    ) -> str:
        """
        Computes the UDF definition query string

        Parameters
        ----------
        udf_name : str
            UDF name
        output_table_names : Optional[List[str]]
            Names of tables returned by UDF

        Returns
        -------
        str
            UDF definition query string
        """
        if output_table_names is not None:
            main_table_name, *sec_table_names = output_table_names
        else:
            main_table_name, sec_table_names = None, None

        builder = UdfDefinitionBuilder(
            funcparts=self.funcparts,
            input_args=self.udf_args,
            output_types=self.output_types,
            smpc_used=self.smpc_used,
        )
        definition = builder.build_udf_definition(
            udf_name, sec_table_names, self.request_id
        )

        # XXX Ugly hack. This is needed because, when SMPC is on, a UDF might
        # produce multiple tables, even if the len(return_types) == 1! In that
        # case, only one table can be returned using a return statement, and
        # the rest are returned using loopback queries. Hence we might need the
        # main_output_table_name for the UDF definition. The reason that this
        # cannot be passed as an argument, and needs to be a template sub
        # instead, is that the method `get_main_return_stmt_template` of
        # `SMPCSecureTransferType` needs to comply with the API of
        # `SecureTransferType` which takes no arguments.
        if main_table_name is not None:
            subs = {"main_output_table_name": main_table_name}
            definition = Template(definition).safe_substitute(subs)

        # XXX and another hack
        definition = definition.replace("$min_row_count", str(self.min_row_count))

        return definition

    def get_exec_stmt(self, udf_name: str, output_table_names: List[str]) -> str:
        """
        Computes UDF execution query

        The execution query is a SELECT statement wrapped inside an INSERT statement.

        Parameters
        ----------
        udf_name : str
            UDF name
        output_table_names : Optional[List[str]]
            Names of tables returned by UDF

        Returns
        -------
        str
            UDF execution query
        """
        table_args = get_items_of_type(TableArg, mapping=self.udf_args)
        main_output_type, *_ = self.output_types
        main_table_name, *_ = output_table_names
        builder = UdfExecStmtBuildfer(table_args)
        return builder.build_exec_stmt(udf_name, main_table_name)

    def get_results(self, output_table_names: List[str]) -> List[UDFGenResult]:
        """
        Computes UDF result objects

        UDF results are represented by instances of the UDFGenResult class.

        Parameters
        ----------
        output_table_names : Optional[List[str]]
            Names of tables returned by UDF

        Returns
        -------
        List[UDFGenResult]
            UDF results
        """
        builder = UdfResultBuilder(self.output_types, self.smpc_used)
        results = builder.build_results(output_table_names)
        return results

    def _get_udf_args(self, flowargs, flowkwargs):
        converter = FlowArgsToUdfArgsConverter()
        args, kwargs = converter.convert(flowargs, flowkwargs, smpc=self.smpc_used)

        builder = UdfArgsBuilder(funcparts=self.funcparts)
        udf_args = builder.build_args(args, kwargs)

        return udf_args


def copy_types_from_udfargs(udfargs: Dict[str, UDFArgument]) -> Dict[str, InputType]:
    return {name: deepcopy(arg.type) for name, arg in udfargs.items()}


def cast_secure_transfers_for_smpc(output_types):
    # There are two flavors of secure transfer types. SecureTransferType, used
    # when the SMPC mechanism is off and SMPCSecureTransferType, used when it
    # is on. At the time a UDF is defined there is no knowledge if it will run
    # with or without the SMPC cluster, hence we need to be able to choose one
    # or the other behaviour at the time of the UDF execution. This is done by
    # casting SecureTransferType to SMPCSecureTransferType when the SMPC
    # mechanism is on.
    for i, output_type in enumerate(output_types):
        if isinstance(output_type, SecureTransferType):
            output_types[i] = SMPCSecureTransferType.cast(output_types[i])


class FlowArgsToUdfArgsConverter:
    """Convenience class for converting arguments from algorithm flow format to
    UDF generator format"""

    def convert(
        self,
        flowargs: List[FlowUdfArg],
        flowkwargs: Dict[str, FlowUdfArg],
        smpc: bool = False,
    ):
        udf_posargs = [self._convert_flowarg(arg, smpc) for arg in flowargs]
        udf_keywordargs = {
            name: self._convert_flowarg(arg, smpc) for name, arg in flowkwargs.items()
        }
        return udf_posargs, udf_keywordargs

    def _convert_flowarg(self, arg: FlowUdfArg, smpc) -> UDFArgument:
        if isinstance(arg, TableInfo):
            return self._convert_tableinfo(arg, smpc)
        if isinstance(arg, SMPCTablesInfo):
            if not smpc:
                raise UDFBadCall("SMPC is not used, so SMPCTablesInfo cannot be used.")
            return self._convert_smpc_tableinfo(arg)
        return LiteralArg(value=arg)

    def _convert_tableinfo(self, table_info: TableInfo, smpc):

        if self._is_transfertype_schema(table_info.schema_.columns):
            return TransferArg(table_name=table_info.name)

        if self._is_secure_transfer_type_schema(table_info.schema_.columns):
            if smpc:
                raise UDFBadCall(
                    "When smpc is used SecureTransferArg should not be used, "
                    "only SMPCSecureTransferArg."
                )
            return SecureTransferArg(table_name=table_info.name)

        if self._is_statetype_schema(table_info.schema_.columns):
            if table_info.type_ == DBTableType.REMOTE:
                raise UDFBadCall("Usage of state is only allowed on local tables.")
            return StateArg(table_name=table_info.name)

        if self._is_tensor_schema(table_info.schema_.columns):
            return self._get_tensor_arg_from_table_info(table_info)

        relation_schema = self._convert_table_schema_to_relation_schema(
            table_info.schema_.columns
        )
        return RelationArg(table_name=table_info.name, schema=relation_schema)

    @staticmethod
    def _get_tensor_arg_from_table_info(table_info):
        ndims = sum(
            1 for col in table_info.schema_.columns if col.name.startswith("dim")
        )
        valcol = next(col for col in table_info.schema_.columns if col.name == "val")
        dtype = valcol.dtype
        return TensorArg(table_name=table_info.name, dtype=dtype, ndims=ndims)

    @staticmethod
    def _is_tensor_schema(schema):
        colnames = [col.name for col in schema]
        if "val" in colnames and any(cname.startswith("dim") for cname in colnames):
            return True
        return False

    @staticmethod
    def _is_transfertype_schema(schema):
        schema = [(col.name, col.dtype) for col in schema]
        return all(column in schema for column in TransferType().schema)

    @staticmethod
    def _is_secure_transfer_type_schema(schema):
        schema = [(col.name, col.dtype) for col in schema]
        return all(column in schema for column in SecureTransferType().schema)

    @staticmethod
    def _is_statetype_schema(schema):
        schema = [(col.name, col.dtype) for col in schema]
        return all(column in schema for column in StateType().schema)

    @staticmethod
    def _convert_table_schema_to_relation_schema(table_schema):
        return [(c.name, c.dtype) for c in table_schema]

    def _convert_smpc_tableinfo(self, smpc_udf_input: SMPCTablesInfo):
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


class UdfArgsBuilder:
    """Builder class for UDF arguments

    This class encapsulates various loosely related procedures, necessary to
    prepare the UDF arguments before they are used by `UdfGenerator`."""

    def __init__(self, funcparts: FunctionParts) -> None:
        self.funcparts = funcparts

    def build_args(self, args, kwargs) -> Dict[str, UDFArgument]:
        udf_args = merge_args_and_kwargs(
            param_names=self.funcparts.sig.parameters.keys(),
            args=args,
            kwargs=kwargs,
        )

        self._prepare_logger_arg(udf_args)
        self._prepare_placeholder_args(udf_args)

        self._resolve_merge_table_args(udf_args)

        self._validate_arg_names(udf_args)
        self._validate_arg_types(udf_args)

        return udf_args

    def _prepare_logger_arg(self, args) -> None:
        # Check logger_param_name argument is not given and if not, create it.
        if self.funcparts.logger_param_name:
            if self.funcparts.logger_param_name in args.keys():
                raise UDFBadCall(
                    "No argument should be provided for "
                    f"'UDFLoggerType' parameter: '{self.funcparts.logger_param_name}'"
                )
            args[self.funcparts.logger_param_name] = UDFLoggerArg()

    def _prepare_placeholder_args(self, args) -> None:
        placeholders = get_items_of_type(PlaceholderType, self.funcparts.sig.parameters)
        if placeholders:
            args.update(
                {
                    name: PlaceholderArg(type=placeholder)
                    for name, placeholder in placeholders.items()
                }
            )

    def _resolve_merge_table_args(self, udf_args: Dict[str, UDFArgument]) -> None:
        """MergeTableTypes have the same schema as the tables that they are merging.
        The UDFArgument always contains the initial table type and must be resolved
        to a MergeTableType, if needed, based on the function parts."""

        def is_merge_tensor(arg, argname, exp_types):
            is_tensor = isinstance(arg, TensorArg)
            return is_tensor and isinstance(exp_types[argname], MergeTensorType)

        def is_merge_transfer(arg, argname, exp_types):
            is_transfer = isinstance(arg, TransferArg)
            return is_transfer and isinstance(exp_types[argname], MergeTransferType)

        for argname, arg in udf_args.items():
            if is_merge_tensor(arg, argname, self.funcparts.table_input_types):
                tensor_type = arg.type
                arg.type = merge_tensor(
                    dtype=tensor_type.dtype, ndims=tensor_type.ndims
                )
            if is_merge_transfer(arg, argname, self.funcparts.table_input_types):
                udf_args[argname].type = merge_transfer()

    def _validate_arg_names(
        self,
        udf_args: Dict[str, UDFArgument],
    ) -> None:
        """Validates that the names of the udf arguments are the expected ones,
        based on the udf's formal parameters."""
        if udf_args.keys() != self.funcparts.sig.parameters.keys():
            raise UDFBadCall(
                f"UDF argument names do not match UDF parameter names: "
                f"{udf_args.keys()}, {self.funcparts.sig.parameters.keys()}."
            )

    def _validate_arg_types(
        self,
        udf_args: Dict[str, UDFArgument],
    ) -> None:
        """Validates that the types of the udf arguments are the expected ones,
        based on the udf's formal parameter types."""
        expected_tables_types = self.funcparts.table_input_types
        expected_literal_types = self.funcparts.literal_input_types

        table_args = get_items_of_type(TableArg, udf_args)
        smpc_args = get_items_of_type(SMPCSecureTransferArg, udf_args)
        literal_args = get_items_of_type(LiteralArg, udf_args)
        for argname, arg in table_args.items():
            if not isinstance(arg.type, type(expected_tables_types[argname])):
                raise UDFBadCall(
                    f"Argument {argname} should be of type "
                    f"{expected_tables_types[argname]}. Type provided: {arg.type}"
                )
        for argname, arg in smpc_args.items():
            if not isinstance(arg.type, type(expected_tables_types[argname])):
                raise UDFBadCall(
                    f"Argument {argname} should be of type "
                    f"{expected_tables_types[argname]}. Type provided: {arg.type}"
                )
        for argname, arg in literal_args.items():
            if not isinstance(arg.type, type(expected_literal_types[argname])):
                raise UDFBadCall(
                    f"Argument {argname} should be of type "
                    f"{expected_literal_types[argname]}. Type provided: {arg.type}"
                )


class UdfDefinitionBuilder:
    """Builder class for the UDF definition query string"""

    def __init__(
        self,
        funcparts: FunctionParts,
        input_args: Dict[str, UDFArgument],
        output_types: List[OutputType],
        smpc_used: bool,
    ) -> Template:
        self.funcparts = funcparts
        self.input_args = input_args
        self.output_types = output_types
        self.smpc_used = smpc_used

        self.main_output_type, *self.sec_output_types = output_types
        self.main_return_name, *self.sec_return_names = funcparts.return_names

    def build_udf_definition(
        self,
        udf_name: str,
        sec_output_table_names: Optional[List[str]],
        request_id: str,
    ):
        if sec_output_table_names is None:
            sec_output_table_names = []
        header = self._build_header(udf_name)
        if self.smpc_used:
            body = self._build_body_smpc(udf_name, sec_output_table_names, request_id)
        else:
            body = self._build_body(udf_name, sec_output_table_names, request_id)
        udf_definition = UDFDefinition(
            header=header,
            body=body,
        )
        return udf_definition.compile()

    @functools.cached_property
    def _table_args(self):
        return get_items_of_type(TableArg, mapping=self.input_args)

    @functools.cached_property
    def _smpc_args(self):
        return get_items_of_type(SMPCSecureTransferArg, mapping=self.input_args)

    @functools.cached_property
    def _literal_args(self):
        return get_items_of_type(LiteralArg, mapping=self.input_args)

    def _logger_arg(self, udf_name, request_id):
        logger_arg_: Optional[str, UDFLoggerArg] = None
        logger_param = self.funcparts.logger_param_name
        if logger_param:
            arg = self.input_args[logger_param]
            arg.udf_name = udf_name
            arg.request_id = request_id
            logger_arg_ = (logger_param, self.input_args[logger_param])
        return logger_arg_

    @functools.cached_property
    def _placeholder_args(self):
        return get_items_of_type(PlaceholderArg, mapping=self.input_args)

    def _build_header(self, udf_name):
        return UDFHeader(
            udfname=udf_name,
            table_args=self._table_args,
            return_type=self.main_output_type,
        )

    def _build_body(self, udf_name, sec_output_table_names, request_id):
        return UDFBody(
            table_args=self._table_args,
            literal_args=self._literal_args,
            logger_arg=self._logger_arg(udf_name, request_id),
            placeholder_args=self._placeholder_args,
            statements=self.funcparts.body_statements,
            main_return_name=self.main_return_name,
            main_return_type=self.main_output_type,
            sec_return_names=self.sec_return_names,
            sec_return_types=self.sec_output_types,
            sec_output_table_names=sec_output_table_names,
        )

    def _build_body_smpc(self, udf_name, sec_output_table_names, request_id):
        return UDFBodySMPC(
            table_args=self._table_args,
            smpc_args=self._smpc_args,
            literal_args=self._literal_args,
            logger_arg=self._logger_arg(udf_name, request_id),
            placeholder_args=self._placeholder_args,
            statements=self.funcparts.body_statements,
            main_return_name=self.main_return_name,
            main_return_type=self.main_output_type,
            sec_return_names=self.sec_return_names,
            sec_return_types=self.sec_output_types,
            sec_output_table_names=sec_output_table_names,
        )


class UdfExecStmtBuildfer:
    """Builder class for the UDF SELECT query string"""

    def __init__(self, table_args: Dict[str, TableArg]):
        self.table_args = table_args

    def build_exec_stmt(self, udf_name: str, main_table_name: str) -> str:
        tensors = self._make_table_ast(self.table_args, arg_type=TensorArg)
        relations = self._make_table_ast(self.table_args, arg_type=RelationArg)
        tables = tensors or relations
        columns = [column for table in tables for column in table.columns.values()]
        if tensors:
            where_clause = self._make_tensors_where_clause(tensors)
        elif relations:
            where_clause = self._make_relations_where_clause(relations)
        else:
            where_clause = None
        subselect = Select(columns, tables, where_clause) if tables else None
        func = TableFunction(name=udf_name, subquery=subselect)
        select = Select([StarColumn()], [func])
        insert = Insert(table=main_table_name, values=select)
        return insert.compile()

    @staticmethod
    def _make_table_ast(table_args, arg_type):
        return [
            Table(name=table.table_name, columns=table.column_names())
            for table in get_items_of_type(arg_type, table_args).values()
        ]

    @staticmethod
    def _make_tensors_where_clause(tensors):
        head, *tail = tensors
        where_clause = [
            head.c[colname] == table.c[colname]
            for table in tail
            for colname in head.columns.keys()
            if colname.startswith("dim")
        ]
        return where_clause

    @staticmethod
    def _make_relations_where_clause(relations):
        head, *tail = relations
        where_clause = [
            head.c[colname] == table.c[colname]
            for table in tail
            for colname in head.columns.keys()
            if colname == "row_id"
        ]
        return where_clause


class UdfResultBuilder:
    """Builder class for the UDF result objects"""

    def __init__(self, output_types: List[OutputType], smpc_used: bool = False) -> None:
        self.output_types = output_types
        self.smpc_used = smpc_used

    def build_results(self, output_table_names: List[str]) -> List[UDFGenResult]:
        main_table_name, *sec_table_names = output_table_names
        main_output_type, *sec_output_types = self.output_types

        udf_outputs = [self._make_result(main_output_type, main_table_name)]
        udf_outputs += [
            self._make_result(output_type, table_name)
            for table_name, output_type in zip(sec_table_names, sec_output_types)
        ]
        return udf_outputs

    def _make_result(
        self,
        output_type: OutputType,
        table_name: str,
    ) -> UDFGenResult:
        if isinstance(output_type, SecureTransferType) and self.smpc_used:
            return self._make_smpc_result(output_type, table_name)
        else:
            return self._make_table_result(output_type, table_name)

    @staticmethod
    def _make_table_result(
        output_type: OutputType,
        table_name: str,
    ) -> UDFGenResult:
        create = CreateTable(table_name, output_type.schema).compile()
        return UDFGenTableResult(
            table_name=table_name,
            table_schema=output_type.schema,
            create_query=create,
        )

    def _make_smpc_result(
        self,
        output_type: SecureTransferType,
        table_name_prefix: str,
    ) -> UDFGenSMPCResult:
        placeholders = get_smpc_tablename_placeholders(table_name_prefix)
        template_ph, sum_op_ph, min_op_ph, max_op_ph = placeholders

        template = self._make_table_result(output_type, template_ph)

        sum_op, min_op, max_op = None, None, None
        if output_type.sum_op:
            sum_op = self._make_table_result(output_type, sum_op_ph)
        if output_type.min_op:
            min_op = self._make_table_result(output_type, min_op_ph)
        if output_type.max_op:
            max_op = self._make_table_result(output_type, max_op_ph)

        return UDFGenSMPCResult(
            template=template,
            sum_op_values=sum_op,
            min_op_values=min_op,
            max_op_values=max_op,
        )

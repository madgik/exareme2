from __future__ import annotations
import ast
from collections import OrderedDict
from dataclasses import dataclass
import inspect
from itertools import groupby
from operator import attrgetter
import os
from string import Template
from textwrap import indent
from textwrap import dedent
from typing import Any
from typing import TypeVar

import astor

from mipengine.algorithms import UDF_REGISTRY
from mipengine.algorithms import TableT
from mipengine.algorithms import TensorT
from mipengine.algorithms import LoopbackTableT
from mipengine.algorithms import LiteralParameterT
from mipengine.algorithms import ScalarT
import mipengine.node.udfgen.patched
from mipengine.node.udfgen.udfparams import Table
from mipengine.node.udfgen.udfparams import LiteralParameter
from mipengine.node.udfgen.udfparams import LoopbackTable
from mipengine.node.udfgen.udfparams import Tensor
from mipengine.node.udfgen.udfparams import Scalar
from mipengine.node.udfgen.udfparams import SQLTYPES


CREATE_OR_REPLACE = "CREATE OR REPLACE"
FUNCTION = "FUNCTION"
RETURNS = "RETURNS"
LANGUAGE_PYTHON = "LANGUAGE PYTHON"
BEGIN = "{"
IMPORTS = [  # TODO solve imports problem
    "import pandas as pd",
    "from scipy.special import expit",
    "from udfio import as_tensor_table",
    "from udfio import from_tensor_table",
    "from udfio import as_relational_table",
]
END = "}"

SELECT = "SELECT"
STAR = " * "
FROM = "FROM"
WHERE = "WHERE"

PRFX = " " * 4
SEP = ", "
LN = os.linesep
SEPLN = SEP + LN
ANDLN = " AND" + LN

STR_TO_TYPE = {
    "int": int,
    "float": float,
    "float64": float,
    "real": float,
    "str": str,
    "text": str,
}

UDFGEN_REGISTRY = {}


@dataclass
class ColumnInfo:
    name: str
    dtype: str


@dataclass
class TableInfo:
    name: str
    nrows: int
    schema: list[ColumnInfo]

    @property
    def shape(self):
        return self.nrows, len(self.schema)


LiteralValue = Any
UdfArgument = TypeVar("UdfArgument", TableInfo, LiteralValue)


def generate_udf_application_steps(
    func_name: str,
    udf_name: str,
    positional_args: list[UdfArgument],
    keyword_args: dict[str, UdfArgument],
) -> tuple[str, str]:
    udf_def = generate_udf(func_name, udf_name, positional_args, keyword_args)
    udf_sel = generate_udf_select_stmt(
        func_name, udf_name, positional_args, keyword_args
    )
    return udf_def, udf_sel


def generate_udf(
    func_name: str,
    udf_name: str,
    positional_args: list[UdfArgument],
    keyword_args: dict[str, UdfArgument],
) -> str:
    """
    Generates definitions in MonetDB Python UDFs from Python functions which
    have been properly annotated using types found in
    mipengine.algorithms.iotypes.

    Parameters
    ----------
        func_name: str
            Name of function from which to generate UDF.
        udf_name: str
            Name to use in UDF definition.
        positional_args: list[dict]
            Positional parameter info objects.
        keyword_args: dict[str, dict]
            Keyword parameter info objects.

    Returns
    -------
        str
            Multiline string with MonetDB Python UDF definition.
    """
    generator = get_generator(func_name)
    args = [create_input_parameter(arg) for arg in positional_args]
    kwargs = {name: create_input_parameter(arg) for name, arg in keyword_args.items()}
    return generator.to_sql(udf_name, *args, **kwargs)


def get_generator(func_name):
    global UDFGEN_REGISTRY
    if func_name in UDFGEN_REGISTRY:
        return UDFGEN_REGISTRY[func_name]
    func = UDF_REGISTRY[func_name]
    validate_type_hints(func)
    gen = UDFGenerator(func)
    UDFGEN_REGISTRY[func_name] = gen
    return gen


def create_input_parameter(parameter):
    if isinstance(parameter, TableInfo):
        return create_input_table(parameter)
    else:
        return LiteralParameter(parameter)


def create_input_table(table):
    shape = table.shape
    dtype = STR_TO_TYPE[table.schema[0].dtype]
    if not all(col.dtype == table.schema[0].dtype for col in table.schema):
        raise TypeError("Can't have different types in columns.")  # TODO yes can
    return Table(dtype, shape=shape)


class UDFGenerator:
    _table_tpl = "return as_relational_table({return_name})"
    _tensor_tpl = "return as_tensor_table({return_name})"  # TODO should coerce to np.array before returning
    _df_def_tpl = "{name} = pd.DataFrame({{n: _columns[n] for n in {colnames}}})"
    _tens_def_tpl = "{name} = from_tensor_table({{n: _columns[n] for n in {colnames}}})"
    _reindex_tpl = "{name}.reindex(_columns['row_id'])"
    _loopbk_call_tpl = '{name} = _conn.execute("SELECT * FROM {lbname}")'

    def __init__(self, func):
        self.func = func
        self.name = func.__name__
        self.code = inspect.getsource(func)
        self.tree = ast.parse(self.code)
        self.body = self._get_body()
        self.return_name = self._get_return_name()
        self.return_type = func.__annotations__["return"]
        self.signature = inspect.signature(func)
        self.tableparams = [
            name
            for name, param in self.signature.parameters.items()
            if param.annotation == TableT
        ]
        self.tensorparams = [
            name
            for name, param in self.signature.parameters.items()
            if param.annotation == TensorT
        ]
        self.literalparams = [
            name
            for name, param in self.signature.parameters.items()
            if param.annotation == LiteralParameterT
        ]
        self.loopbackparams = [
            name
            for name, param in self.signature.parameters.items()
            if param.annotation == LoopbackTableT
        ]

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def _get_return_name(self):
        body_stmts = self.tree.body[0].body  # type: ignore
        ret_stmt = next(s for s in body_stmts if isinstance(s, ast.Return))
        if isinstance(ret_stmt.value, ast.Name):
            ret_name = ret_stmt.value.id
        else:
            raise NotImplementedError("No expressions in return stmt, for now")
        return ret_name

    def _get_body(self):
        body_ast = self.tree.body[0].body  # type: ignore
        statemets = [stmt for stmt in body_ast if type(stmt) != ast.Return]
        body = dedent("".join(astor.to_source(stmt) for stmt in statemets))
        return body

    def to_sql(self, udf_name, *args, **kwargs):
        def verify_types(args, kwargs):
            allowed = (Table, Tensor, LiteralParameter, LoopbackTable)
            args_are_allowed = [
                type(ar) in allowed for ar in args + tuple(kwargs.values())
            ]
            if not all(args_are_allowed):
                msg = f"Can't convert to SQL: arguments must have types in {allowed}"
                raise TypeError(msg)

        def gather_inputs(args, kwargs):
            argnames = [
                name
                for name in self.signature.parameters.keys()
                if name not in kwargs.keys()
            ]
            all_args = dict(**dict(zip(argnames, args)), **kwargs)
            inputs = OrderedDict(
                **{name: all_args[name] for name in self.signature.parameters.keys()}
            )
            return inputs

        def make_declaration_input_params(inputs):
            input_params = [
                input_param.as_sql_parameters(name)
                for name, input_param in inputs.items()
                if name in self.tableparams + self.tensorparams
            ]
            input_params = SEP.join(input_params)
            return input_params

        def get_return_statement(return_type):
            if return_type == TableT:
                return_stmt = self._table_tpl.format(return_name=self.return_name)
            elif return_type == TensorT:
                return_stmt = self._tensor_tpl.format(return_name=self.return_name)
            else:
                return_stmt = f"return {self.return_name}\n"
            return return_stmt

        def get_output_type(output):
            if type(output) == Table:
                output_expr = output.as_sql_return_type(self.return_name)
            elif type(output) == Tensor:  # TODO override as_sql_return_type
                # Not used: func never returns Tensor
                output_expr = output.as_sql_return_type(self.return_name)
            elif type(output) == Scalar:
                output_expr = output.as_sql_return_type()  # Is this used?
            else:
                output_expr = SQLTYPES[type(output)]
            return output_expr

        def gen_table_def_code(inputs):
            table_defs = []
            for name in self.tableparams:
                table = inputs[name]
                tabcolnames = [f"{name}{i}" for i in range(table.ncols)]
                table_defs += [self._df_def_tpl.format(name=name, colnames=tabcolnames)]
            for name in self.tensorparams:
                tensor = inputs[name]
                tabcolnames = [f"{name}{i}" for i in range(tensor.ncols)]
                table_defs += [
                    self._tens_def_tpl.format(name=name, colnames=tabcolnames)
                ]
            table_defs = LN.join(table_defs)
            return table_defs

        def gen_loopback_calls_code(inputs):
            loopback_calls = []
            for name in self.loopbackparams:
                lb = inputs[name]
                loopback_calls += [
                    self._loopbk_call_tpl.format(name=name, lbname=lb.name)
                ]
            loopback_calls = LN.join(loopback_calls)
            return loopback_calls

        def gen_literal_def_code(inputs):
            literal_defs = []
            for name in self.literalparams:
                ltr = inputs[name]
                literal_defs += [f"{name} = {ltr.value}"]
            literal_defs = LN.join(literal_defs)
            return literal_defs

        verify_types(args, kwargs)
        inputs = gather_inputs(args, kwargs)
        output = self(*args, **kwargs)
        input_params = make_declaration_input_params(inputs)
        udf_signature = f"{udf_name}({input_params})"
        return_stmt = get_return_statement(self.return_type)
        output_type = get_output_type(output)
        table_defs = gen_table_def_code(inputs)
        loopback_calls = gen_loopback_calls_code(inputs)
        literal_defs = gen_literal_def_code(inputs)

        funcdef = [
            CREATE_OR_REPLACE,
            FUNCTION,
            udf_signature,
            RETURNS,
            output_type,
            LANGUAGE_PYTHON,
            BEGIN,
        ]
        funcdef.extend([indent(line, PRFX) for line in IMPORTS])
        funcdef += [indent(table_defs, PRFX)] if table_defs else []
        funcdef += [indent(loopback_calls, PRFX)] if loopback_calls else []
        funcdef += [indent(literal_defs, PRFX)] if literal_defs else []
        funcdef += [indent(self.body, PRFX)] if self.body else []
        funcdef += [indent(return_stmt, PRFX)]
        funcdef += [END]

        funcdef = remove_blank_lines(funcdef)

        return LN.join(funcdef)


def validate_type_hints(func):
    allowed_types = (TableT, TensorT, LiteralParameterT, LoopbackTableT)
    sig = inspect.signature(func)
    argnames = sig.parameters.keys()
    type_hints = func.__annotations__
    argument_hints = dict(**type_hints)
    del argument_hints["return"]
    if any(type_hints.get(arg, None) not in allowed_types for arg in argnames):
        raise TypeError("Function is not properly annotated as a Monet UDF")
    if type_hints.get("return", None) not in allowed_types + (ScalarT,):
        raise TypeError("Function is not properly annotated as a Monet UDF")
    if TableT in argument_hints.values() and TensorT in argument_hints.values():
        raise TypeError("Can't have both TableT and TensorT in udf annotation")


def generate_udf_select_stmt(
    func_name: str,
    udf_name: str,
    positional_args: list[UdfArgument],
    keyword_args: dict[str, UdfArgument],  # XXX not used for now
) -> str:
    """
    Generates select statement for calling MonetDB Python UDFs.

    Parameters
    ----------
        func_name: str
            Name of function from which to generate UDF.
        udf_name: str
            Name to use in UDF definition.
        positional_args: list[dict]
            Positional parameter info objects.
        keyword_args: dict[str, dict]
            Keyword parameter info objects.

    Returns
    -------
        str
            Multiline string with select statemet calling a MonetDB Python UDF.
    """
    gen = UDFGEN_REGISTRY.get(func_name, None) or get_generator(func_name)
    parameters = gen.signature.parameters
    return_hint = gen.signature.return_annotation
    type_hints = {name: parameter.annotation for name, parameter in parameters.items()}

    if TableT in type_hints.values():
        main_input_type = TableT
    elif TensorT in type_hints.values():
        main_input_type = TensorT
    else:
        raise NotImplementedError

    tables = [
        arg
        for arg, hint in zip(positional_args, type_hints.values())
        if hint == main_input_type
    ]
    table_names = [table.name for table in tables]
    table_schemas = [[col.name for col in table.schema] for table in tables]

    udf_arguments = [
        f"{table}.{column}"
        for table, columns in zip(table_names, table_schemas)
        for column in columns
    ]
    udf_call_args = prettify(SEP.join(udf_arguments))

    from_subexpr = prettify(SEP.join(table_names))

    if TableT in type_hints.values():
        head_table, *tail_tables = table_names
        join_on = [f"{head_table}.row_id={table}.row_id" for table in tail_tables]
        where_subexpr = ANDLN.join(join_on)
    elif TensorT in type_hints.values():
        if not all_equal(tables, attrgetter("shape")):
            raise TypeError("Can't have tensors of different sizes in python udf")
        ndims = len(tables[0].schema) - 1
        all_dims = [f"dim{i}" for i in range(ndims)]
        tensor_dims = [[f"{name}.{dim}" for dim in all_dims] for name in table_names]
        head_dims, *tail_dims = tensor_dims
        join_on = [
            ANDLN.join([f"{a}={b}" for a, b in zip(head_dims, other)])
            for other in tail_dims
        ]
        where_subexpr = ANDLN.join(join_on)
    else:
        raise NotImplementedError

    if return_hint not in (TableT, TensorT):
        select_lines = [SELECT]
        select_lines.append(indent(udf_name + parens(udf_call_args), PRFX))
        select_lines.append(FROM)
        select_lines.append(indent(from_subexpr, PRFX))
        if where_subexpr:
            select_lines.append(WHERE)
            select_lines.append(indent(where_subexpr, PRFX))
        select_stmt = LN.join(select_lines)

    else:
        subquery_lines = [SELECT]
        subquery_lines.append(indent(udf_call_args, PRFX))
        subquery_lines.append(FROM)
        subquery_lines.append(indent(from_subexpr, PRFX))
        if where_subexpr:
            subquery_lines.append(WHERE)
            subquery_lines.append(indent(where_subexpr, PRFX))
        subquery = LN.join(subquery_lines)
        select_lines = [SELECT + STAR]
        select_lines.append(FROM)
        select_lines.append(indent(udf_name + parens(parens(subquery)), PRFX))
        select_stmt = LN.join(select_lines)

    return select_stmt


def all_equal(iterable, func=None):
    """Returns True if all elements in iterable are equal, False otherwise. If
    func is passed a new iterable is used by mapping func to iterable."""
    itr = map(func, iterable) if func else iterable
    g = groupby(itr)
    return next(g, True) and not next(g, False)


def remove_blank_lines(text):
    try:
        text = text.splitlines()
    except AttributeError:
        pass
    return map(lambda s: s.replace(LN, ""), text)


def prettify(lst_expr):
    if len(lst_expr) > 80:
        return LN + indent(lst_expr.replace(SEP, SEPLN), PRFX) + LN
    return lst_expr


def parens(expr):
    if LN not in expr and len(expr) <= 78:
        return "(" + expr + ")"
    else:
        return "(\n" + expr + "\n)"


if __name__ == "__main__":
    t1 = TableInfo(
        name="tab1",
        nrows=10,
        schema=[
            ColumnInfo("a", "int"),
            ColumnInfo("b", "int"),
            ColumnInfo("c", "int"),
            ColumnInfo("d", "int"),
        ],
    )
    t2 = TableInfo(
        name="tab2",
        nrows=10,
        schema=[
            ColumnInfo("A", "int"),
            ColumnInfo("B", "int"),
            ColumnInfo("C", "int"),
            ColumnInfo("D", "int"),
        ],
    )
    udfstr = generate_udf("demo.func", "example", [t1], {})
    print(udfstr)

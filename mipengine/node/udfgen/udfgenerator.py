from __future__ import annotations
import ast
from collections import OrderedDict
from dataclasses import dataclass
import functools
import inspect
from itertools import groupby
from operator import attrgetter
from operator import concat
import os
from string import Template
from textwrap import indent
from textwrap import dedent
from textwrap import fill
from typing import Any
from typing import TypeVar
from typing import get_type_hints
from typing import get_origin
from typing import get_args

import astor
import numpy as np

from mipengine.algorithms import UDF_REGISTRY
from mipengine.algorithms import TableT
from mipengine.algorithms import RelationT
from mipengine.algorithms import LoopbackRelationFromSchemaT
from mipengine.algorithms import RelationFromSchemaT
from mipengine.algorithms import LoopbackRelationT
from mipengine.algorithms import TensorT
from mipengine.algorithms import LoopbackTensorT
from mipengine.algorithms import LiteralParameterT
from mipengine.algorithms import ScalarT
import mipengine.node.udfgen.patched


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

SQLTYPES = {
    "int": "INT",
    int: "BIGINT",
    float: "DOUBLE",
    str: "TEXT",
    np.int32: "INT",
    np.int64: "BIGINT",
    np.float32: "FLOAT",
    np.float64: "DOUBLE",
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
        return self.nrows, self.ncolumns

    @property
    def ncolumns(self):
        return len(self.schema)


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
    return generator.generate_code(udf_name, *positional_args, **keyword_args)


def get_generator(func_name):
    global UDFGEN_REGISTRY
    if func_name in UDFGEN_REGISTRY:
        return UDFGEN_REGISTRY[func_name]
    func = UDF_REGISTRY[func_name]
    gen = UDFCodeGenerator(func)
    UDFGEN_REGISTRY[func_name] = gen
    return gen


class FunctionAnalyzer:
    def __init__(self, func) -> None:
        self.name = func.__name__
        code = inspect.getsource(func)
        self.tree = ast.parse(code)
        self.body = self.get_body()
        self.signature = inspect.signature(func)
        self.type_hints = get_type_hints(func)
        self.return_name = self.get_return_name()
        self.return_hint = self.signature.return_annotation
        self.return_type = get_origin(self.return_hint)
        self.return_typevars = get_args(self.return_hint)
        self.parameters = self.signature.parameters
        self.parameter_hints = [param.annotation for param in self.parameters.values()]
        self.parameter_types = [get_origin(param) for param in self.parameter_hints]
        self.parameter_typevars = [get_args(param) for param in self.parameter_hints]

    def get_return_name(self):
        body_stmts = self.tree.body[0].body  # type: ignore
        ret_stmt = next(s for s in body_stmts if isinstance(s, ast.Return))
        if isinstance(ret_stmt.value, ast.Name):
            ret_name = ret_stmt.value.id
        else:
            raise NotImplementedError("No expressions in return stmt, for now")
        return ret_name

    def get_body(self):
        body_ast = self.tree.body[0].body  # type: ignore
        statemets = [stmt for stmt in body_ast if type(stmt) != ast.Return]
        body = dedent("".join(astor.to_source(stmt) for stmt in statemets))
        return body


class UDFCodeGenerator:
    _table_tpl = "return as_relational_table({return_name})"
    _tensor_tpl = "return as_tensor_table(numpy.array({return_name}))"
    _df_def_tpl = "{name} = pd.DataFrame({{n: _columns[n] for n in {colnames}}})"
    _tens_def_tpl = "{name} = from_tensor_table({{n: _columns[n] for n in {colnames}}})"
    _reindex_tpl = "{name}.reindex(_columns['row_id'])"
    _loopbk_call_tpl = '{name} = _conn.execute("SELECT * FROM {lbname}")'

    def __init__(self, func) -> None:
        self.func = func
        self._funcparts = FunctionAnalyzer(func)
        self._validate_type_hints()
        self._group_parameters()

    def _validate_type_hints(self):
        allowed_types = (
            RelationT,
            LoopbackRelationT,
            RelationFromSchemaT,
            LoopbackRelationFromSchemaT,
            TensorT,
            LoopbackTensorT,
            LiteralParameterT,
            ScalarT,
        )
        sig = self._funcparts.signature
        argnames = sig.parameters.keys()
        type_hints = self._funcparts.type_hints
        argument_hints = dict(**type_hints)
        del argument_hints["return"]
        if any(
            type_hints.get(arg, None).__origin__ not in allowed_types
            for arg in argnames
        ):
            raise TypeError("Function parameters are not properly annotated.")
        if get_origin(type_hints.get("return", None)) not in allowed_types + (ScalarT,):
            raise TypeError("Function return type is not allowed.")
        if RelationT in argument_hints.values() and TensorT in argument_hints.values():
            raise TypeError("Can't have both RelationT and TensorT in udf annotation.")

    def _group_parameters(self):
        self.relation_params = [
            name
            for name, param in self._funcparts.signature.parameters.items()
            if get_origin(param.annotation) in (RelationT, RelationFromSchemaT)
        ]
        self.lbrelation_params = [
            name
            for name, param in self._funcparts.signature.parameters.items()
            if get_origin(param.annotation) == LoopbackRelationT
        ]
        self.tensor_params = [
            name
            for name, param in self._funcparts.signature.parameters.items()
            if get_origin(param.annotation) == TensorT
        ]
        self.lbtensor_params = [
            name
            for name, param in self._funcparts.signature.parameters.items()
            if get_origin(param.annotation) == LoopbackTensorT
        ]
        self.literal_params = [
            name
            for name, param in self._funcparts.signature.parameters.items()
            if get_origin(param.annotation) == LiteralParameterT
        ]

    def generate_code(self, udf_name, *args, **kwargs):
        self._validate_arg_types(args, kwargs)
        inputs = self._gather_inputs(args, kwargs)
        if self._output_type_is_known():
            output_type = self._compute_known_output_type(args, kwargs)
        else:
            # output = self.func(*args, **kwargs)
            raise NotImplementedError
        input_params = self._make_signature(inputs)
        udf_signature = f"{udf_name}({input_params})"
        return_stmt = self._get_return_statement()
        return_name = self._funcparts.return_name
        sql_return_type = self._get_return_type(output_type)
        table_defs = self._gen_table_defs(inputs)
        loopback_calls = self._gen_loopback_calls(inputs)
        literal_defs = self._gen_literal_defs(inputs)

        funcdef = [
            CREATE_OR_REPLACE,
            FUNCTION,
            udf_signature,
            RETURNS,
            sql_return_type,
            LANGUAGE_PYTHON,
            BEGIN,
        ]
        funcdef.extend([indent(line, PRFX) for line in IMPORTS])
        funcdef.extend([indent(table_def, PRFX) for table_def in table_defs])
        funcdef.extend([indent(lbcall, PRFX) for lbcall in loopback_calls])
        funcdef.extend([indent(literal_def, PRFX) for literal_def in literal_defs])
        funcdef.append(indent(self._funcparts.body, PRFX))
        funcdef.append(indent(return_stmt, PRFX))
        funcdef.append(END)

        funcdef = remove_blank_lines(funcdef)

        return LN.join(funcdef)

    def _validate_arg_types(self, args, kwargs):
        parameters = self._funcparts.signature.parameters
        for arg, param in zip(args, parameters.values()):
            paramtype = get_origin(param.annotation)
            if not isinstance(arg, TableInfo) and not issubclass(paramtype, TableT):
                TypeError("Expected type {paramtype}, got type {type(arg)}.")
        for name, arg in kwargs.items():
            paramtype = get_origin(parameters[name].annotation)
            if not isinstance(arg, TableInfo) and not issubclass(paramtype, TableT):
                TypeError("Expected type {paramtype}, got type {type(arg)}.")

    def _gather_inputs(self, args, kwargs):
        argnames = [
            name
            for name in self._funcparts.signature.parameters.keys()
            if name not in kwargs.keys()
        ]
        all_args = dict(**dict(zip(argnames, args)), **kwargs)
        if len(all_args) != len(self._funcparts.signature.parameters):
            msg = format_multiline_msg(
                """Arguments passed to UDFCodeGenerator do not match
                corresponding number formal parameters.
                """
            )
            raise ValueError(msg)
        inputs = OrderedDict(
            **{
                name: all_args[name]
                for name in self._funcparts.signature.parameters.keys()
            }
        )
        return inputs

    def _output_type_is_known(self) -> bool:
        return_typevarset = set(self._funcparts.return_typevars)
        parameter_typevarset = set(flatten(self._funcparts.parameter_typevars))
        return not (return_typevarset - parameter_typevarset)

    def _compute_known_output_type(self, args, kwargs):
        output_type = self._funcparts.return_type
        return_typeparams = self._funcparts.return_type.__parameters__
        return_typevars = self._funcparts.return_typevars
        return_typevar_map = dict(zip(return_typeparams, return_typevars))
        parameter_types = self._funcparts.parameter_types
        parameter_typevars = self._funcparts.parameter_typevars
        output_args = []
        for typevar_name, typevar_val in return_typevar_map.items():
            typevar_idx = [typevar_val in ptv for ptv in parameter_typevars].index(True)
            output_args.append(self._get_typevar_val(args[typevar_idx], typevar_name))
        output = output_type(*output_args)
        return output

    @staticmethod
    def _get_typevar_val(arg, typevar):
        if typevar.__name__ == "NColumns":
            return arg.shape[1]
        elif typevar.__name__ == "NRows":
            return arg.shape[0]
        elif typevar.__name__ == "DType":
            return arg.dtype
        elif typevar.__name__ == "Schema":
            return arg.schema
        else:
            raise KeyError(f"Unknown type var {typevar}")

    def _make_signature(self, inputs):
        parameter_types = self._funcparts.parameter_types
        input_params = [
            param_type(input_.schema).as_sql_parameters(name)
            for (name, input_), param_type in zip(inputs.items(), parameter_types)
            if name in self.relation_params + self.tensor_params
        ]
        input_params = SEP.join(input_params)
        return input_params

    def _get_return_statement(self):
        return_type = self._funcparts.return_hint
        return_name = self._funcparts.return_name
        if return_type == RelationT:
            return_stmt = self._table_tpl.format(return_name=return_name)
        elif return_type == TensorT:
            return_stmt = self._tensor_tpl.format(return_name=return_name)
        else:
            return_stmt = f"return {return_name}\n"
        return return_stmt

    def _get_return_type(self, output_type):
        return_name = self._funcparts.return_name
        return output_type.as_sql_return_type(return_name)

    def _gen_table_defs(self, inputs):
        table_defs = []
        for name in self.relation_params:
            table = inputs[name]
            tabcolnames = [f"{name}_{col.name}" for col in table.schema]
            table_defs += [self._df_def_tpl.format(name=name, colnames=tabcolnames)]
        for name in self.tensor_params:
            tensor = inputs[name]
            tabcolnames = [f"{name}_{col.name}" for col in tensor.schema]
            table_defs += [self._tens_def_tpl.format(name=name, colnames=tabcolnames)]
        return table_defs

    def _gen_loopback_calls(self, inputs):
        loopback_calls = []
        for name in self.lbrelation_params:
            lb = inputs[name]
            loopback_calls += [self._loopbk_call_tpl.format(name=name, lbname=lb.name)]
        return loopback_calls

    def _gen_literal_defs(self, inputs):
        literal_defs = []
        for name in self.literal_params:
            ltr = inputs[name]
            literal_defs += [f"{name} = {ltr.value}"]
        return literal_defs


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
    parameters = gen._funcparts.signature.parameters
    return_hint = gen._funcparts.signature.return_annotation
    type_hints = {name: parameter.annotation for name, parameter in parameters.items()}

    if RelationT in type_hints.values():
        main_input_type = RelationT
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

    if RelationT in type_hints.values():
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

    if return_hint not in (RelationT, TensorT):
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


def format_multiline_msg(msg):
    msglines = list(map(dedent, msg.splitlines()))
    msg = LN.join(msglines)
    return fill(msg, width=50)


flatten = functools.partial(functools.reduce, concat)

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
    udfstr = generate_udf("demo.func", "example", [t1, t2], {})
    print(udfstr)

from __future__ import annotations
from abc import ABC
from abc import abstractproperty
import ast
from collections import OrderedDict
import functools
import inspect
import itertools
import operator
import os
import string
from textwrap import indent
from textwrap import dedent
from textwrap import fill
from typing import Any
from typing import NamedTuple
from typing import TypeVar
from typing import get_type_hints
from typing import get_origin
from typing import get_args

import astor
import numpy as np

from mipengine.algorithms import UDF_REGISTRY
from mipengine.algorithms import TableT
from mipengine.algorithms import RelationT
from mipengine.algorithms import LoopbackRelationT
from mipengine.algorithms import TensorT
from mipengine.algorithms import LoopbackTensorT
from mipengine.algorithms import LiteralParameterT
from mipengine.algorithms import ScalarT


CREATE_OR_REPLACE = "CREATE OR REPLACE"
FUNCTION = "FUNCTION"
RETURNS = "RETURNS"
LANGUAGE_PYTHON = "LANGUAGE PYTHON"
BEGIN = "{"
IMPORTS = [  # TODO solve imports problem
    "import pandas as pd",
    "from udfio import as_tensor_table",
    "from udfio import from_tensor_table",
    "from udfio import as_relational_table",
]
END = "}"
DROP_IF_EXISTS = "DROP TABLE IF EXISTS "
CREATE_TABLE = "CREATE TABLE "
SELECT = "SELECT"
STAR = " * "
FROM = "FROM"
WHERE = "WHERE"
AS = " AS "

PRFX = " " * 4
SEP = ", "
LN = os.linesep
SEPLN = SEP + LN
ANDLN = " AND" + LN
SCOLON = ";"

UDFGEN_REGISTRY = {}


class ColumnInfo(NamedTuple):
    name: str
    dtype: str


class TableInfo(NamedTuple):
    name: str
    schema: list[ColumnInfo]
    # nrows: int  # not used?


LiteralValue = Any
UdfArgument = TypeVar("UdfArgument", TableInfo, LiteralValue)


class TableV(ABC):
    def __repr__(self) -> str:
        cls = type(self).__name__
        attrs = self.__dict__
        attrs_rep = str(attrs).replace("'", "").replace(": ", "=").strip("{}")
        rep = f"{cls}({attrs_rep})"
        return rep

    @property
    def ncolumns(self):
        return len(self.schema)

    @property
    def columns(self):
        return [f"{self.name}_{name}" for name, _ in self.schema]

    @abstractproperty
    def schema(self):
        raise NotImplementedError

    def as_udf_signature(self):
        raise NotImplementedError

    def as_udf_return_type(self):
        return_signature = SEP.join([f"{name} {dtype}" for name, dtype in self.schema])
        return f"TABLE({return_signature})"


class RelationV(TableV):
    def __init__(self, name, schema) -> None:
        self.name = name
        self._schema = schema

    def as_udf_signature(self):
        return SEP.join([f"{self.name}_{name} {dtype}" for name, dtype in self.schema])

    @property
    def schema(self):
        return self._schema


class TensorV(TableV):
    def __init__(self, name, ndims, dtype) -> None:
        self.name = name
        self.ndims = ndims
        self.dtype = dtype

    def as_udf_signature(self):
        signature = [f"{self.name}_dim{d} INT" for d in range(self.ndims)]
        signature.append(f"{self.name}_val {self.dtype}")
        return SEP.join(signature)

    @property
    def schema(self):
        schema = [ColumnInfo(f"dim{d}", "INT") for d in range(self.ndims)]
        schema.append(ColumnInfo("val", str(self.dtype)))
        return schema


def generate_udf_application_queries(
    func_name: str,
    positional_args: list[UdfArgument],
    keyword_args: dict[str, UdfArgument],
) -> tuple[string.Template, string.Template]:
    if keyword_args:
        msg = "Calling with keyword arguments is not implemented yet."
        raise NotImplementedError(msg)
    udf_def = generate_udf_def(func_name, positional_args, keyword_args)
    udf_sel = generate_udf_select_stmt(func_name, positional_args, keyword_args)
    udf_create_table = generate_udf_create_table(udf_sel)
    return string.Template(udf_def), string.Template(udf_create_table)


def generate_udf_def(
    func_name: str,
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
    parameter_typess = generator.funcparts.parameter_types
    args = []
    for arg, (pname, ptype) in zip(positional_args, parameter_typess.items()):
        if ptype == RelationT:
            args.append(RelationV(name=pname, schema=arg.schema))
        elif ptype == TensorT:
            try:
                dtype = next(col.dtype for col in arg.schema if col.name == "val")
            except StopIteration:
                raise TypeError("TableInfo doesn't have tensor-like schema.")
            args.append(TensorV(name=pname, ndims=len(arg.schema) - 1, dtype=dtype))
        else:
            args.append(arg)
    return generator.generate_code(*args, **keyword_args)


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
        self.return_name = self.get_return_name()

        signature = inspect.signature(func)
        self.analyze_return_type(signature)
        self.analyze_parameter_types(signature)

    def get_return_name(self) -> str:
        body_stmts = self.tree.body[0].body  # type: ignore
        ret_stmt = next(s for s in body_stmts if isinstance(s, ast.Return))
        if isinstance(ret_stmt.value, ast.Name):
            ret_name = ret_stmt.value.id
        else:
            raise NotImplementedError("No expressions in return stmt, for now")
        return ret_name

    def get_body(self) -> str:
        body_ast = self.tree.body[0].body  # type: ignore
        statemets = [stmt for stmt in body_ast if type(stmt) != ast.Return]
        body = dedent("".join(astor.to_source(stmt) for stmt in statemets))
        return body

    def analyze_return_type(self, signature) -> None:
        return_hint = signature.return_annotation
        self.return_type = get_origin(return_hint)
        ret_typevarnames = map(typevar_to_attr_name, self.return_type.__parameters__)
        self.return_typevars = OrderedDict(zip(ret_typevarnames, get_args(return_hint)))

    def analyze_parameter_types(self, signature) -> None:
        parameters = signature.parameters
        pnames = parameters.keys()
        parameter_hints = [param.annotation for param in parameters.values()]
        parameter_types = [get_origin(param) for param in parameter_hints]
        self.parameter_types = OrderedDict(zip(pnames, parameter_types))
        parameter_typevars = [get_args(param) for param in parameter_hints]
        param_typevar_values = dict(zip(pnames, parameter_typevars))
        self.parameter_typevars = OrderedDict()
        for pname, ptype in self.parameter_types.items():
            typevar_names = map(typevar_to_attr_name, ptype.__parameters__)
            typevar_vals = param_typevar_values[pname]
            self.parameter_typevars[pname] = dict(zip(typevar_names, typevar_vals))


class UDFCodeGenerator:
    _table_tpl = "return as_relational_table({return_name})"
    _tensor_tpl = "return as_tensor_table(numpy.array({return_name}))"
    _df_def_tpl = "{name} = pd.DataFrame({{n: _columns[n] for n in {colnames}}})"
    _tens_def_tpl = "{name} = from_tensor_table({{n: _columns[n] for n in {colnames}}})"
    _reindex_tpl = "{name}.reindex(_columns['row_id'])"
    _loopbk_call_tpl = '{name} = _conn.execute("SELECT * FROM {lbname}")'

    def __init__(self, func) -> None:
        self.func = func
        self.funcparts = FunctionAnalyzer(func)
        # self._validate_type_hints()
        self._group_parameters()

    def _group_parameters(self):
        parameter_types = self.funcparts.parameter_types
        self.relation_params = [
            pname for pname, ptype in parameter_types.items() if ptype == RelationT
        ]
        self.lbrelation_params = [
            pname
            for pname, ptype in parameter_types.items()
            if ptype == LoopbackRelationT
        ]
        self.tensor_params = [
            pname for pname, ptype in parameter_types.items() if ptype == TensorT
        ]
        self.lbtensor_params = [
            pname
            for pname, ptype in parameter_types.items()
            if ptype == LoopbackTensorT
        ]
        self.literal_params = [
            pname
            for pname, ptype in parameter_types.items()
            if ptype == LiteralParameterT
        ]

    def generate_code(self, *args, **kwargs):
        udf_name = "$udf_name"
        inputs = self._gather_inputs(args, kwargs)
        self._validate_input_types(inputs)
        if self._output_type_is_known():
            output = self._compute_known_output(inputs)
        else:
            # output = self.func(*args, **kwargs)
            raise NotImplementedError
        input_params = self._make_signature(inputs)
        udf_signature = f"{udf_name}({input_params})"
        return_stmt = self._get_return_statement()
        return_name = self.funcparts.return_name
        sql_return_type = output.as_udf_return_type()
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
        funcdef.append(indent(self.funcparts.body, PRFX))
        funcdef.append(indent(return_stmt, PRFX))
        funcdef.append(END + SCOLON)

        funcdef = remove_blank_lines(funcdef)

        return LN.join(funcdef)

    def _gather_inputs(self, args, kwargs):
        pnames = self.funcparts.parameter_types.keys()
        if len(args) + len(kwargs) != len(pnames):
            msg = format_multiline_msg(
                """Arguments passed to UDFCodeGenerator do not match
                corresponding number formal parameters.
                """
            )
            raise ValueError(msg)
        argnames = [name for name in pnames if name not in kwargs.keys()]
        argsmap = OrderedDict(zip(argnames, args))
        inputs = OrderedDict(
            {pname: argsmap.get(pname, None) or kwargs.get(pname) for pname in pnames}
        )
        return inputs

    def _validate_input_types(self, inputs):
        parameter_types = self.funcparts.parameter_types
        for pname, ptype in parameter_types.items():
            arg = inputs[pname]
            if not isinstance(arg, TableInfo) and not issubclass(ptype, TableT):
                TypeError("Expected type {ptype}, got type {type(arg)}.")

    def _output_type_is_known(self) -> bool:
        return_typevarset = set(self.funcparts.return_typevars.values())
        parameter_typevars = self.funcparts.parameter_typevars
        parameter_typevarset = set(
            flatten([list(_.values()) for _ in parameter_typevars.values()])
        )
        return not (return_typevarset - parameter_typevarset)

    def _compute_known_output(self, inputs):
        # We compute the output object by assembling attributes from input
        # objects and using one of the constructors (RelationV, TensorV).
        # The algorithm works as follows:
        # For every type variable in return type hint:
        #     For type variables in every input parameter:
        #         If return type variable value present in parameter type variables:
        #             Add mapping (type variable name -> parameter name) to dictionary
        # For every typevar_name, pname pair of dictionary:
        #     Get attribute of name typevar_name from parameter of name pname
        # Use attributes with constructor to build output object

        return_typevars = self.funcparts.return_typevars
        parameter_typevars = self.funcparts.parameter_typevars

        # Find args holding a type var value that matches a return type var value
        found_args = dict()
        for ret_typevarname, ret_typevarval in return_typevars.items():
            for pname, ptypevars in parameter_typevars.items():
                if ret_typevarval in ptypevars.values():
                    found_args[ret_typevarname] = pname
                    break

        return_name = self.funcparts.return_name
        output_type = self.funcparts.return_type

        # Get from parameter object the attributes described by relevant type var names
        output_constructor_args = dict()
        for typevar_name, pname in found_args.items():
            cons_arg = getattr(inputs[found_args[typevar_name]], typevar_name)
            output_constructor_args[typevar_name] = cons_arg

        # Get constructor
        if output_type == RelationT:
            output_constructor = RelationV
        elif output_type == TensorT:
            output_constructor = TensorV
        else:
            msg = "If output type is not RelationT or TensorT we should be here."
            raise ValueError(msg)

        output = output_constructor(name=return_name, **output_constructor_args)
        return output

    def _make_signature(self, inputs):
        parameter_types = self.funcparts.parameter_types
        input_params = [
            input_.as_udf_signature()
            for (name, input_) in inputs.items()
            if name in self.relation_params + self.tensor_params
        ]
        input_params = SEP.join(input_params)
        return input_params

    def _get_return_statement(self):
        return_type = self.funcparts.return_type
        return_name = self.funcparts.return_name
        if return_type == RelationT:
            return_stmt = self._table_tpl.format(return_name=return_name)
        elif return_type == TensorT:
            return_stmt = self._tensor_tpl.format(return_name=return_name)
        else:
            return_stmt = f"return {return_name}"
        return return_stmt

    def _get_return_type(self, output_type):
        return_name = self.funcparts.return_name
        return output_type.as_sql_return_type(return_name)

    def _gen_table_defs(self, inputs):
        table_defs = []
        for name in self.relation_params:
            table = inputs[name]
            table_defs += [self._df_def_tpl.format(name=name, colnames=table.columns)]
        for name in self.tensor_params:
            table = inputs[name]
            table_defs += [self._tens_def_tpl.format(name=name, colnames=table.columns)]
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
    udf_name = "$udf_name"
    gen = UDFGEN_REGISTRY.get(func_name, None) or get_generator(func_name)
    parameter_types = gen.funcparts.parameter_types
    return_type = gen.funcparts.return_type

    # Validate we don't have both TableT types in parameters
    if (
        RelationT in parameter_types.values()
        and not TensorT in parameter_types.values()
    ):
        main_input_type = RelationT
    elif (
        TensorT in parameter_types.values()
        and not RelationT in parameter_types.values()
    ):
        main_input_type = TensorT
    else:
        raise NotImplementedError

    # Filter tables involved in select statement
    tables = [
        arg
        for arg, type in zip(positional_args, parameter_types.values())
        if type == main_input_type
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

    if RelationT == main_input_type:
        head_table, *tail_tables = table_names
        join_on = [f"{head_table}.row_id={table}.row_id" for table in tail_tables]
        where_subexpr = ANDLN.join(join_on)
    elif TensorT == main_input_type:
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

    if return_type not in (RelationT, TensorT):
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


def generate_udf_create_table(udf_select_stmt):
    table_name = "$table_name"
    query = [DROP_IF_EXISTS + table_name + SCOLON]
    query.append(CREATE_TABLE + table_name + AS + parens(udf_select_stmt) + SCOLON)
    return LN.join(query)


def all_equal(iterable, func=None):
    """Returns True if all elements in iterable are equal, False otherwise. If
    func is passed a new iterable is used by mapping func to iterable."""
    itr = map(func, iterable) if func else iterable
    g = itertools.groupby(itr)
    return next(g, True) and not next(g, False)


def any_(iterable, predicate=operator.truth):
    for x in iterable:
        if predicate(x):
            return True
    return False


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
        return "(\n" + indent(expr, PRFX) + "\n)"


def format_multiline_msg(msg):
    msglines = list(map(dedent, msg.splitlines()))
    msg = LN.join(msglines)
    return fill(msg, width=50)


def typevar_to_attr_name(typevar):
    # TODO hack, find a better way
    return typevar.__name__.lower()


flatten = functools.partial(functools.reduce, operator.concat)

if __name__ == "__main__":
    t1 = TableInfo(
        name="tab1",
        schema=[
            ColumnInfo("a", "int"),
            ColumnInfo("b", "int"),
            ColumnInfo("c", "int"),
            ColumnInfo("d", "int"),
        ],
    )
    t2 = TableInfo(
        name="tab2",
        schema=[
            ColumnInfo("A", "int"),
            ColumnInfo("B", "int"),
            ColumnInfo("C", "int"),
            ColumnInfo("D", "int"),
        ],
    )
    udf, query = generate_udf_application_queries("demo.func", [t1, t2], {})
    print(udf.substitute(udf_name="yaya"))
    print(query.substitute(udf_name="yaya", table_name="bababa"))

    print("-" * 50)
    t1 = TableInfo(
        name="tab1",
        schema=[
            ColumnInfo("dim0", "int"),
            ColumnInfo("dim1", "int"),
            ColumnInfo("dim2", "int"),
            ColumnInfo("val", "float"),
        ],
    )
    t2 = TableInfo(
        name="tab2",
        schema=[
            ColumnInfo("dim0", "int"),
            ColumnInfo("dim1", "int"),
            ColumnInfo("val", "float"),
        ],
    )
    udf, query = generate_udf_application_queries("demo.tensor3", [t1, t2, t1], {})
    print(udf.substitute(udf_name="yaya"))
    print(query.substitute(udf_name="yaya", table_name="bababa"))

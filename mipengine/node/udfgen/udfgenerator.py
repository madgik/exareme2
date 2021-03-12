from __future__ import annotations
from abc import ABC
from abc import abstractmethod
from abc import abstractproperty
import ast
from collections import OrderedDict
from dataclasses import dataclass
from dataclasses import field
import functools
import inspect
import itertools
import operator
import os
import re
import string
from textwrap import indent
from textwrap import dedent
from textwrap import fill
from typing import Any
from typing import Optional
from typing import NamedTuple
from typing import TypeVar
from typing import Generic
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
from mipengine.node.udfgen.sql_linalg import SQL_LINALG_QUERIES


CREATE_OR_REPLACE = "CREATE OR REPLACE"
FUNCTION = "FUNCTION"
RETURNS = "RETURNS"
LANGUAGE_PYTHON = "LANGUAGE PYTHON"
BEGIN = "{"
IMPORTS = [  # TODO solve imports problem
    "import pandas as pd",
    "import udfio",
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


PY2SQL_TYPES: dict[type, str] = {int: "INT", float: "FLOAT", str: "TEXT"}


class ColumnInfo(NamedTuple):
    name: str
    dtype: str


@dataclass
class TableInfo:
    name: str
    schema: list[ColumnInfo]
    nrows: Optional[int] = field(default=None)

    def __post_init__(self):
        if any(map(str.isupper, self.name)):
            msg = f"Uppercase letters are not allowed in table names: {self.name}."
            raise ValueError(msg)
        for cname, _ in self.schema:
            if any(map(str.isupper, cname)):
                msg = f"Uppercase letters are not allowed in column names: {cname}."
                raise ValueError(msg)


LiteralValue = Any
UdfArgument = TypeVar("UdfArgument", TableInfo, LiteralValue)


class UdfIOValue(ABC):
    """Objects of this class mirror objects of subclasses of UdfIOType. They
    represent instantiations of the types represented by UdfIOType. They are used
    as data object placeholders within the UDF generation/calling mechanism."""

    def __repr__(self) -> str:
        cls = type(self).__name__
        attrs = self.__dict__
        attrs_rep = str(attrs).replace("'", "").replace(": ", "=").strip("{}")
        rep = f"{cls}({attrs_rep})"
        return rep

    @abstractmethod
    def as_udf_return_type(self):
        raise NotImplementedError


class TableV(UdfIOValue, ABC):
    name: str

    @property
    def ncolumns(self):
        return len(self.schema)

    @property
    def columns(self):
        return [f"{self.name}_{name}" for name, _ in self.schema]

    @abstractproperty
    def schema(self):
        raise NotImplementedError

    @abstractmethod
    def as_udf_signature(self):
        raise NotImplementedError

    def as_udf_return_type(self) -> str:
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

    @classmethod
    def from_table_info(cls, table_info):
        try:
            dtype = next(col.dtype for col in table_info.schema if col.name == "val")
        except StopIteration:
            raise TypeError("TableInfo doesn't have tensor-like schema.")
        return cls(name=table_info.name, ndims=len(table_info.schema) - 1, dtype=dtype)


class ScalarV(UdfIOValue):
    def __init__(self, dtype: type) -> None:
        self.dtype = dtype

    def as_udf_return_type(self) -> str:
        return PY2SQL_TYPES[self.dtype]


TYPES2CONS = {RelationT: RelationV, TensorT: TensorV, ScalarT: ScalarV}


def generate_udf_application_queries(
    func_name: str,
    positional_args: list[UdfArgument],
    keyword_args: dict[str, UdfArgument],
) -> tuple[string.Template, string.Template]:
    if keyword_args:
        msg = "Calling with keyword arguments is not implemented yet."
        raise NotImplementedError(msg)

    # --> Hack for calling hard-coded UDFs
    if func_name.startswith("sql"):
        udf_gen_func = SQL_LINALG_QUERIES[func_name]
        udf_gen_func_args = [
            t.name if isinstance(t, TableInfo) else t for t in positional_args
        ]
        udf_def, udf_sel = udf_gen_func(*udf_gen_func_args)
        udf_create_table = generate_udf_create_table(udf_sel)
        return string.Template(udf_def), string.Template(udf_create_table)
    # <--
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
    parameter_types = generator.funcparts.parameter_types
    if (pn := len(parameter_types)) != (an := len(positional_args) + len(keyword_args)):
        raise ValueError(f"{func_name} expected {pn} arguments, {an} where given.")
    args = []
    for arg, (pname, ptype) in zip(positional_args, parameter_types.items()):
        if ptype == RelationT and isinstance(arg, TableInfo):
            args.append(RelationV(name=pname, schema=arg.schema))
        elif ptype == LoopbackRelationT and isinstance(arg, TableInfo):
            args.append(RelationV(name=pname, schema=arg.schema))
        elif ptype == TensorT and isinstance(arg, TableInfo):
            args.append(TensorV.from_table_info(arg))
        elif ptype == LoopbackTensorT and isinstance(arg, TableInfo):
            args.append(TensorV.from_table_info(arg))
        elif ptype == LiteralParameterT and not isinstance(arg, TableInfo):
            args.append(arg)
        else:
            raise TypeError("Arguments given do not match UDF formal parameter types")

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
        self._return_hint = signature.return_annotation
        self._parameter_hints = OrderedDict()
        parameters = signature.parameters
        for pname, param in parameters.items():
            self._parameter_hints[pname] = param.annotation

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

    @staticmethod
    def _get_typehint_type(type_hint):
        if type_hint.typevars_are_bound:
            return type(type_hint)
        else:
            return get_origin(type_hint)

    @staticmethod
    def _get_typehint_free_typevars(type_hint):
        if type_hint.typevars_are_bound:
            return dict()
        else:
            type_param_names = (
                typevar_to_attr_name(name)
                for name in get_origin(type_hint).__parameters__
            )
            return dict(zip(type_param_names, get_args(type_hint)))

    @staticmethod
    def _get_typehint_bound_typevars(type_hint):
        if type_hint.typevars_are_bound:
            type_param_names = (
                typevar_to_attr_name(name) for name in type(type_hint).__parameters__
            )
            return {
                type_param_name: getattr(type_hint, type_param_name)
                for type_param_name in type_param_names
            }
        else:
            return dict()

    @functools.cached_property
    def return_type(self):
        return self._get_typehint_type(self._return_hint)

    @functools.cached_property
    def return_free_typevars(self):
        return self._get_typehint_free_typevars(self._return_hint)

    @functools.cached_property
    def return_bound_typevars(self):
        return self._get_typehint_bound_typevars(self._return_hint)

    @functools.cached_property
    def parameter_types(self):
        parameter_types = OrderedDict()
        for pname, param in self._parameter_hints.items():
            parameter_types[pname] = self._get_typehint_type(param)
        return parameter_types

    @functools.cached_property
    def parameter_free_typevars(self):
        parameter_free_typevars = OrderedDict()
        for pname, param in self._parameter_hints.items():
            parameter_free_typevars[pname] = self._get_typehint_free_typevars(param)
        return parameter_free_typevars

    @functools.cached_property
    def parameter_bound_typevars(self):
        parameter_bound_typevars = OrderedDict()
        for pname, param in self._parameter_hints.items():
            parameter_bound_typevars[pname] = self._get_typehint_bound_typevars(param)
        return parameter_bound_typevars

    def get_return_obj_constructor(self):
        return TYPES2CONS[self.return_type]  # type: ignore


class UDFCodeGenerator:
    ret_tab = "return {return_name}"
    ret_tens = "return udfio.as_tensor_table(numpy.array({return_name}))"
    def_df = "{name} = pd.DataFrame({{n: _columns[n] for n in {colnames}}})"
    def_tens = "{name} = udfio.from_tensor_table({{n:_columns[n] for n in {colnames}}})"
    call_loopbk = '{name} = _conn.execute("SELECT * FROM {lbname}")'

    def __init__(self, func) -> None:
        self.func = func
        self.funcparts = FunctionAnalyzer(func)
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
        if self._return_typevars_are_bound():
            return_obj = self._build_return_obj()
        elif self._return_obj_has_known_attrs():
            return_obj = self._build_return_obj_from_inputs(inputs)
        else:
            # output = self.func(*args, **kwargs)
            raise NotImplementedError
        input_params = self._make_parameter_signature(inputs)
        udf_signature = f"{udf_name}({input_params})"
        return_stmt = self._get_return_statement()
        return_name = self.funcparts.return_name
        sql_return_type = return_obj.as_udf_return_type()
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

    def _return_typevars_are_bound(self):
        return self.funcparts._return_hint.typevars_are_bound

    def _build_return_obj(self):
        return_cons = self.funcparts.get_return_obj_constructor()
        return_args = self.funcparts.return_bound_typevars
        if 'name' in inspect.signature(return_cons).parameters:
            return_args['name'] = self.funcparts.return_name
        return return_cons(**return_args)

    def _return_obj_has_known_attrs(self) -> bool:
        return_typevarset = set(self.funcparts.return_free_typevars.values())
        parameter_typevars = self.funcparts.parameter_free_typevars.values()
        parameter_typevarset = set(flatten([_.values() for _ in parameter_typevars]))
        return not (return_typevarset - parameter_typevarset)

    def _build_return_obj_from_inputs(self, inputs):
        # We compute the output object by assembling attributes from input
        # objects and using one of the constructors (RelationV, TensorV).
        # The algorithm works as follows:
        # For every free type variable in return type hint:
        #     For free type variables in every input parameter:
        #         If return type variable value present in parameter type variables:
        #             Add mapping (type variable name -> parameter name) to dictionary
        # For every typevar_name, pname pair of dictionary:
        #     Get attribute of name typevar_name from parameter of name pname
        # Use attributes with constructor to build output object

        return_typevars = self.funcparts.return_free_typevars
        parameter_typevars = self.funcparts.parameter_free_typevars

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
            msg = "If output type is not RelationT or TensorT we shouldn't be here."
            raise ValueError(msg)

        output = output_constructor(name=return_name, **output_constructor_args)
        return output

    def _make_parameter_signature(self, inputs):
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
            return_stmt = self.ret_tab.format(return_name=return_name)
        elif return_type == TensorT:
            return_stmt = self.ret_tens.format(return_name=return_name)
        else:
            return_stmt = f"return {return_name}"
        return return_stmt

    def _gen_table_defs(self, inputs):
        table_defs = []
        for name in self.relation_params:
            table = inputs[name]
            table_defs += [self.def_df.format(name=name, colnames=table.columns)]
        for name in self.tensor_params:
            table = inputs[name]
            table_defs += [self.def_tens.format(name=name, colnames=table.columns)]
        return table_defs

    def _gen_loopback_calls(self, inputs):
        loopback_calls = []
        for name in self.lbrelation_params:
            lb = inputs[name]
            loopback_calls += [self.call_loopbk.format(name=name, lbname=lb.name)]
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

    nodeid_column = " $node_id" + AS + "node_id"

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
        select_lines = [SELECT + nodeid_column + SEP + STAR]
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
    return map(lambda line: re.sub(r"(.*)(\n)$", r"\1", line), text)


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


flatten = itertools.chain.from_iterable

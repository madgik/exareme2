from __future__ import annotations
import ast
from collections import OrderedDict
from functools import lru_cache
import inspect
from string import Template
from textwrap import indent
from textwrap import dedent

import astor

from mipengine.algorithms import UDF_REGISTRY
from mipengine.algorithms import TableT
from mipengine.algorithms import TensorT
from mipengine.algorithms import LoopbackTableT
from mipengine.algorithms import LiteralParameterT
from mipengine.algorithms import ScalarT
from mipengine.node.udfgen.udfparams import Table
from mipengine.node.udfgen.udfparams import LiteralParameter
from mipengine.node.udfgen.udfparams import LoopbackTable
from mipengine.node.udfgen.udfparams import Tensor
from mipengine.node.udfgen.udfparams import Scalar
from mipengine.node.udfgen.udfparams import SQLTYPES

INPUTTABLE = "input_table"
LOOPBACK = "loopback_table"
LITERAL = "literal_parameter"

CREATE_OR_REPLACE = "CREATE OR REPLACE"
FUNCTION = "FUNCTION"
RETURNS = "RETURNS"
LANGUAGE_PYTHON = "LANGUAGE PYTHON"
BEGIN = "{"
IMPORTS = "from mipengine.udfgen import ArrayBundle"
END = "};"


STR_TO_TYPE = {"int": int, "real": float, "text": str}


def generate_udf(
    func_name: str,
    udf_name: str,
    positional_args: list[dict],
    keyword_args: dict[str, dict],
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


@lru_cache
def get_generator(func_name):
    func = UDF_REGISTRY[func_name]
    verify_annotations(func)
    return UDFGenerator(func)


def create_input_parameter(parameter):
    if parameter["type"] == INPUTTABLE:
        return create_input_table(parameter)
    elif parameter["type"] == LOOPBACK:
        return create_loopback_table(parameter)
    elif parameter["type"] == LITERAL:
        return create_literal(parameter)
    else:
        raise TypeError(f"Unknown parameter type {parameter['type']}")


def create_input_table(descr):
    schema = descr["schema"]
    nrows = descr["nrows"]
    ncols = len(schema)
    dtype = STR_TO_TYPE[schema[0]["type"]]
    if not all(col["type"] == schema[0]["type"] for col in schema):
        raise TypeError("Can't have different types in columns.")
    return Table(dtype, shape=(nrows, ncols))


def create_loopback_table(descr):
    name = descr["name"]
    schema = descr["schema"]
    nrows = descr["nrows"]
    ncols = len(schema)
    dtype = STR_TO_TYPE[schema[0]["type"]]
    if not all(col["type"] == schema[0]["type"] for col in schema):
        raise TypeError("Can't have different types in columns.")
    return LoopbackTable(name, dtype, shape=(nrows, ncols))


def create_literal(descr):
    value = descr["value"]
    return LiteralParameter(value)


class UDFGenerator:
    _table_template = Template("return as_relational_table($return_name)")
    _tensor_template = Template("return as_tensor_table($return_name)")

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
        body_stmts = self.tree.body[0].body
        ret_stmt = next(s for s in body_stmts if isinstance(s, ast.Return))
        if isinstance(ret_stmt.value, ast.Name):
            ret_name = ret_stmt.value.id
        else:
            raise NotImplementedError("No expressions in return stmt, for now")
        return ret_name

    def _get_body(self):
        body_ast = self.tree.body[0].body
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
            input_params = ", ".join(input_params)
            return input_params

        def get_return_statement(return_type):
            if return_type == TableT:
                return_stmt = self._table_template.substitute(
                    return_name=self.return_name
                )
            elif return_type == TensorT:
                return_stmt = self._tensor_template.substitute(
                    return_name=self.return_name
                )
            else:
                return_stmt = f"return {self.return_name}\n"
            return return_stmt

        def get_output_type(output):
            if type(output) == Table:
                output_expr = output.as_sql_return_type(self.return_name)
            elif type(output) == Tensor:
                output_expr = output.as_sql_return_type(self.return_name)
            elif type(output) == Scalar:
                output_expr = output.as_sql_return_type()
            elif type(output) == LiteralParameter:
                output_expr = SQLTYPES[type(output.value)]
            else:
                output_expr = SQLTYPES[type(output)]
            return output_expr

        def gen_table_def_code(inputs):
            table_defs = []
            stop = 0
            for name in self.tableparams:
                table = inputs[name]
                start, stop = stop, stop + table.shape[1]
                table_defs += [f"{name} = ArrayBundle(_columns[{start}:{stop}])"]
            for name in self.tensorparams:
                tensor = inputs[name]
                start, stop = stop, stop + tensor.shape[1]
                table_defs += [f"{name} = from_tensor_table(_columns[{start}:{stop}])"]
            table_defs = "\n".join(table_defs)
            return table_defs

        def gen_loopback_calls_code(inputs):
            loopback_calls = []
            for name in self.loopbackparams:
                lpb = inputs[name]
                loopback_calls += [
                    f'{name} = _conn.execute("SELECT * FROM {lpb.name}")'
                ]
            loopback_calls = "\n".join(loopback_calls)
            return loopback_calls

        def gen_literal_def_code(inputs):
            literal_defs = []
            for name in self.literalparams:
                ltr = inputs[name]
                literal_defs += [f"{name} = {ltr.value}"]
            literal_defs = "\n".join(literal_defs)
            return literal_defs

        verify_types(args, kwargs)
        inputs = gather_inputs(args, kwargs)
        output = self(*args, **kwargs)
        input_params = make_declaration_input_params(inputs)
        return_stmt = get_return_statement(self.return_type)
        output_expr = get_output_type(output)
        table_defs = gen_table_def_code(inputs)
        loopback_calls = gen_loopback_calls_code(inputs)
        literal_defs = gen_literal_def_code(inputs)

        prfx = " " * 4

        funcdef = [
            CREATE_OR_REPLACE,
            FUNCTION,
            f"{udf_name}({input_params})",
            RETURNS,
            f"{output_expr}",
            LANGUAGE_PYTHON,
            BEGIN,
            indent(IMPORTS, prfx),
        ]
        funcdef += [indent(table_defs, prfx)] if table_defs else []
        funcdef += [indent(loopback_calls, prfx)] if loopback_calls else []
        funcdef += [indent(literal_defs, prfx)] if literal_defs else []
        funcdef += ["", "    # body", indent(self.body, prfx)] if self.body else []
        funcdef += [indent(return_stmt, prfx)]
        funcdef += [END]

        return "\n".join(funcdef)


def verify_annotations(func):
    allowed_types = (TableT, TensorT, LiteralParameterT, LoopbackTableT)
    sig = inspect.signature(func)
    argnames = sig.parameters.keys()
    annotations = func.__annotations__
    if any(annotations.get(arg, None) not in allowed_types for arg in argnames):
        raise TypeError("Function is not properly annotated as a Monet UDF")
    if annotations.get("return", None) not in allowed_types + (ScalarT,):
        raise TypeError("Function is not properly annotated as a Monet UDF")

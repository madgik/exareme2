import inspect
import ast
from textwrap import indent
from textwrap import dedent
from string import Template

import astor

from worker.udfgen import Table
from worker.udfgen import LiteralParameter
from worker.udfgen import LoopbackTable
from worker.udfgen import Tensor
from worker.udfgen.udfparams import SQLTYPES


UDF_REGISTER = {}


class UDFGenerator:
    _udftemplate = Template(
        """CREATE OR REPLACE FUNCTION
$func_name($input_params)
RETURNS
$output_expr
LANGUAGE PYTHON
{
$table_defs
$loopbacks
$literals

    # method body
$body
$return_stmt
};
"""
    )
    _table_tmpl = Template("return as_relational_table($return_name)")
    _tensor_tmpl = Template("return as_tensor_table($return_name)")

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
            if param.annotation == Table
        ]
        self.literalparams = [
            name
            for name, param in self.signature.parameters.items()
            if param.annotation == LiteralParameter
        ]
        self.loopbackparams = [
            name
            for name, param in self.signature.parameters.items()
            if param.annotation == LoopbackTable
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

    def to_sql(self, *args, **kwargs):
        # verify types
        allowed = (Table, Tensor, LiteralParameter, LoopbackTable)
        args_are_allowed = [type(ar) in allowed for ar in args + tuple(kwargs.values())]
        if not all(args_are_allowed):
            msg = f"Can't convert to SQL: all arguments must have types in {allowed}"
            raise TypeError(msg)

        # get inputs and output
        argnames = [
            name
            for name in self.signature.parameters.keys()
            if name not in kwargs.keys()
        ]
        inputs = dict(**dict(zip(argnames, args)), **kwargs)
        output = self(*args, **kwargs)

        # get input params expression
        input_params = [
            inputs[name].as_sql_parameters(name) for name in self.tableparams
        ]
        input_params = ", ".join(input_params)

        # get return statement
        if type(output) == Table:
            output_expr = output.as_sql_return_declaration(self.return_name)
            return_stmt = self._table_tmpl.substitute(return_name=self.return_name)
        elif type(output) == Tensor:
            output_expr = output.as_sql_return_declaration(self.return_name)
            return_stmt = self._tensor_tmpl.substitute(return_name=self.return_name)
        else:
            output_expr = SQLTYPES[type(output)]
            return_stmt = f"return {self.return_name}\n"

        # gen code for ArrayBundle definitions
        table_defs = []
        stop = 0
        for name in self.tableparams:
            table = inputs[name]
            start, stop = stop, stop + table.shape[1]
            table_defs += [f"{name} = ArrayBundle(_columns[{start}:{stop}])"]
        table_defs = "\n".join(table_defs)

        # gen code for loopback params
        loopback_calls = []
        for name in self.loopbackparams:
            lpb = inputs[name]
            loopback_calls += [f'{name} = _conn.execute("SELECT * FROM {lpb.name}")']
        loopback_calls = "\n".join(loopback_calls)

        # gen code for literal parameters
        literal_defs = []
        for name in self.literalparams:
            ltr = inputs[name]
            literal_defs += [f"{name} = {ltr.value}"]
        literal_defs = "\n".join(literal_defs)

        # output udf code
        prfx = " " * 4
        return self._udftemplate.substitute(
            func_name=self.name,
            input_params=input_params,
            output_expr=output_expr,
            table_defs=indent(table_defs, prfx),
            loopbacks=indent(loopback_calls, prfx),
            literals=indent(literal_defs, prfx),
            body=indent(self.body, prfx),
            return_stmt=indent(return_stmt, prfx),
        )


def monet_udf(func):
    global UDF_REGISTER

    verify_annotations(func)

    ugen = UDFGenerator(func)
    UDF_REGISTER[ugen.name] = ugen
    return ugen


def verify_annotations(func):
    allowed_types = (Table, Tensor, LiteralParameter, LoopbackTable)
    sig = inspect.signature(func)
    argnames = sig.parameters.keys()
    annotations = func.__annotations__
    if any(annotations.get(arg, None) not in allowed_types for arg in argnames):
        raise TypeError("Function is not properly annotated as a Monet UDF")
    if annotations.get("return", None) not in allowed_types:
        raise TypeError("Function is not properly annotated as a Monet UDF")


def generate_udf(udf_name, table_schema, table_rows, loopback_tables, literalparams):
    gen = UDF_REGISTER[udf_name]

    table = create_table(table_schema, table_rows)

    loopback_tables = [
        create_table(lp["table_schema"], lp["table_rows"], name)
        for name, lp in loopback_tables.items()
    ]

    literals = [LiteralParameter(literalparams[name]) for name in gen.literalparams]
    return UDF_REGISTER[udf_name].to_sql(table, *loopback_tables, *literals)


def create_table(table_schema, table_rows, name=None):
    ncols = len(table_schema)
    dtype = table_schema[0]["type"]
    if not all(col["type"] == dtype for col in table_schema):
        raise TypeError("Can't have different types in columns yet")
    if name:
        table = LoopbackTable(name, dtype, shape=(table_rows, ncols))
    else:
        table = Table(dtype, shape=(table_rows, ncols))
    return table


# -------------------------------------------------------- #
# Examples                                                 #
# -------------------------------------------------------- #


@monet_udf
def binarize_labels(y: Table, classes: LiteralParameter) -> Table:
    from algorithms.preprocessing import LabelBinarizer

    binarizer = LabelBinarizer()
    binarizer.fit(classes)
    binarized = binarizer.transform(y)
    return binarized


@monet_udf
def zeros(shape: LiteralParameter) -> Tensor:
    import numpy2 as np

    z = np.zeros(shape)
    return z


print(zeros.to_sql(shape=LiteralParameter((2, 3))))

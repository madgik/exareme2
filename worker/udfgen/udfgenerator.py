import inspect
import ast
from textwrap import indent
from textwrap import dedent
from itertools import chain

import numpy as np
import astor

from table import Table

SQLTYPES = {
    int: "BIGINT",
    float: "DOUBLE",
    str: "TEXT",
    np.int32: "INT",
    np.int64: "BIGINT",
    np.float32: "FLOAT",
    np.float64: "DOUBLE",
}


def get_delcaration(func):
    sourcelines = inspect.getsourcelines(func)[0]
    return sourcelines[1]


class UDFGenerator:
    def __init__(self, func):
        self.func = func
        self.name = func.__name__
        self.code = inspect.getsource(func)
        self.tree = ast.parse(self.code)
        self.body = self._get_body()
        self.return_name = self._get_return_name()
        self.signature = inspect.signature(func)

    def __call__(self, *args, **kwargs):
        mocktypes = (Table,)
        result = self.func(*args, **kwargs)
        args_are_mocks = [
            type(arg) in mocktypes for arg in args + tuple(kwargs.values())
        ]
        if all(args_are_mocks):
            argnames = [
                name
                for name in self.signature.parameters.keys()
                if name not in kwargs.keys()
            ]
            inputs = dict(**dict(zip(argnames, args)), **kwargs)
            output = result
            print(self.emit_code(inputs, output))

        return result

    def _get_return_name(self):
        body_stmts = self.tree.body[0].body
        ret_idx = [isinstance(s, ast.Return) for s in body_stmts].index(True)
        ret_stmt = body_stmts[ret_idx]
        if isinstance(ret_stmt.value, ast.Name):
            ret_name = ret_stmt.value.id
        else:
            raise NotImplementedError("No expressions in return stmt, for now")
        return ret_name

    def _get_body(self):
        statemets = [
            stmt for stmt in self.tree.body[0].body if type(stmt) != ast.Return
        ]
        body = dedent("\n".join(astor.to_source(stmt) for stmt in statemets))
        return body

    def emit_code(self, inputs, output):
        tablenames = self.signature.parameters.keys()
        input_types = [
            [
                name + str(_) + " " + SQLTYPES[inputs[name].type]
                for _ in range(inputs[name].shape[1])
            ]
            for name in tablenames
        ]
        input_types = ", ".join(chain(*input_types))

        if isinstance(output, Table):
            output_types = ", ".join(
                [
                    self.return_name + str(_) + " " + SQLTYPES[output.type]
                    for _ in range(output.shape[1])
                ]
            )
            output_type = f"Table({output_types})"
        else:
            output_type = SQLTYPES[type(output)]
        prfx = " " * 4
        code = [f"CREATE OR REPLACE FUNCTION {self.name}({input_types})"]
        code += [f"RETURNS {output_type}"]
        code += ["LANGUAGE PYTHON {"]
        code += [indent("import numpy as np", prfx)]
        code += [indent("from arraybundle import ArrayBundle", prfx)]
        code += [indent("table = ArrayBundle(_columns)\n", prfx)]
        code += [indent("# start method body", prfx)]

        code.extend(indent(self.body, prfx).splitlines())
        if isinstance(output, Table):
            code += [""]
            code += [indent(f"_colrange = range({self.return_name}.shape[1])", prfx)]
            code += [
                indent(
                    f'_names = (f"{self.return_name}_{{i}}" for i in _colrange)', prfx
                )
            ]
            code += [
                indent(
                    f"_result = {{n: c for n, c in zip(_names, {self.return_name})}}",
                    prfx,
                )
            ]
            code += [indent("return _result", prfx)]
        else:
            code += [indent(f"return {self.return_name}", prfx)]
        code += ["};"]
        return "\n".join(code)


def monet_udf(func):
    verify_annotations(func)

    return UDFGenerator(func)


def verify_annotations(func):
    mocktypes = (Table,)
    annotations = func.__annotations__
    return all(tp in mocktypes for tp in annotations.values())


# -------------------------------------------------------- #
# Examples                                                 #
# -------------------------------------------------------- #
@monet_udf
def f(x: Table, y: Table, z: Table):
    return x


x = Table(tp=int, ncolumns=10)
y = Table(tp=float, ncolumns=5)
z = Table(tp=float, ncolumns=2)
f(x, y, z=z)


@monet_udf
def compute_gramian(data: Table):
    gramian = data.T @ data
    return gramian


compute_gramian(Table(tp=int, ncolumns=5))


@monet_udf
def log_table(data: Table):
    result = np.log(data)
    return result


@monet_udf
def ret_one(data: Table):
    result = 1
    return result


ret_one(Table(tp=int, ncolumns=1))

import inspect
import ast
from textwrap import indent
from textwrap import dedent
from itertools import chain
from string import Template

import numpy as np
import astor

from table import Table

UDFTEMPLATE = Template(
    """CREATE OR REPLACE FUNCTION
$func_name($input_params)
RETURNS
$output_expr
LANGUAGE PYTHON
{
    import numpy as np
    from arraybundle import ArrayBundle
$table_defs

    # method body
$body
$return_stmt
};
"""
)

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
        input_params = [
            [
                name + str(_) + " " + SQLTYPES[inputs[name].type]
                for _ in range(inputs[name].shape[1])
            ]
            for name in tablenames
        ]
        input_params = ", ".join(chain(*input_params))

        if isinstance(output, Table):
            output_params = ", ".join(
                [
                    self.return_name + str(_) + " " + SQLTYPES[output.type]
                    for _ in range(output.shape[1])
                ]
            )
            output_expr = f"Table({output_params})"
        else:
            output_expr = SQLTYPES[type(output)]

        if isinstance(output, Table):
            return_stmt = (
                f"_colrange = range({self.return_name}.shape[1])\n"
                + f'_names = (f"{self.return_name}_{{i}}" for i in _colrange)\n'
                + f"_result = {{n: c for n, c in zip(_names, {self.return_name})}}\n"
                + "return _result\n"
                + f"return {self.return_name}\n"
            )
        else:
            return_stmt = f"return {self.return_name}\n"

        table_defs = []
        stop = 0
        for name, table in inputs.items():
            start, stop = stop, stop + table.shape[1]
            table_defs += [f"{name} = ArrayBundle(_columns[{start}:{stop}])"]
        table_defs = "\n".join(table_defs)

        prfx = " " * 4
        subs = dict(
            func_name=self.name,
            input_params=input_params,
            output_expr=output_expr,
            table_defs=indent(table_defs, prfx),
            body=indent(self.body, prfx),
            return_stmt=indent(return_stmt, prfx),
        )
        return UDFTEMPLATE.substitute(subs)


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
def half_table(table: Table):
    ncols = table.shape[1]
    if ncols >= 2:
        result = table  # [:, 0: (ncols // 2)]
    else:
        result = table
    return result


half_table(Table(tp=float, ncolumns=12))


@monet_udf
def ret_one(data: Table):
    result = 1
    return result


ret_one(Table(tp=int, ncolumns=1))

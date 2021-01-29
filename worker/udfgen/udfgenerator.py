import inspect
import ast
from textwrap import indent
from textwrap import dedent
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

RETURNTABLE_TEMPLATE = Template(
    """
_colrange = range(${return_name}.shape[1])
_names = (f"${return_name}_{i}" for i in _colrange)
_result = {n: c for n, c in zip(_names, ${return_name})}
return _result
return ${return_name}"""
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
        statemets = [
            stmt for stmt in self.tree.body[0].body if type(stmt) != ast.Return
        ]
        body = dedent("\n".join(astor.to_source(stmt) for stmt in statemets))
        return body

    def to_sql(self, *args, **kwargs):
        mocktypes = (Table,)
        args_are_mocks = [type(ar) in mocktypes for ar in args + tuple(kwargs.values())]
        if all(args_are_mocks):
            argnames = [
                name
                for name in self.signature.parameters.keys()
                if name not in kwargs.keys()
            ]
            inputs = dict(**dict(zip(argnames, args)), **kwargs)
            output = self(*args, **kwargs)
        else:
            msg = f"Can't convert to SQL: all arguments must have types in {mocktypes}"
            raise TypeError(msg)

        tablenames = self.signature.parameters.keys()
        input_params = [
            f"{name}{_} {SQLTYPES[inputs[name].dtype]}"
            for name in tablenames
            for _ in range(inputs[name].shape[1])
        ]
        input_params = ", ".join(input_params)

        if isinstance(output, Table):
            output_params = [
                f"{self.return_name}{_} {SQLTYPES[output.dtype]}"
                for _ in range(output.shape[1])
            ]
            output_params = ", ".join(output_params)
            output_expr = f"Table({output_params})"
        else:
            output_expr = SQLTYPES[type(output)]

        if isinstance(output, Table):
            return_stmt = RETURNTABLE_TEMPLATE.substitute(
                dict(return_name=self.return_name)
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
    allowed_types = (Table,)
    sig = inspect.signature(func)
    argnames = sig.parameters.keys()
    annotations = func.__annotations__
    if any(annotations.get(arg, None) not in allowed_types for arg in argnames):
        raise TypeError("Function is not properly annotated as a Monet UDF")


# -------------------------------------------------------- #
# Examples                                                 #
# -------------------------------------------------------- #
@monet_udf
def f(x: Table, y: Table, z: Table):
    return x


x = Table(dtype=int, shape=(100, 10))
y = Table(dtype=float, shape=(5, 2))
z = Table(dtype=float, shape=(2,))
f(x, y, z=z)


@monet_udf
def compute_gramian(data: Table):
    gramian = data.T @ data
    return gramian


compute_gramian(Table(dtype=int, shape=(100, 10)))


@monet_udf
def half_table(table: Table):
    ncols = table.shape[1]
    if ncols >= 2:
        result = table[:, 0 : (ncols // 2)]
    else:
        result = table
    return result


half_table(Table(dtype=float, shape=(50, 12)))


@monet_udf
def ret_one(data: Table):
    result = 1
    return result


ret_one(Table(dtype=int, shape=(1,)))

from __future__ import annotations
from itertools import groupby
from operator import attrgetter
from textwrap import indent
from typing import TypeVar

from mipengine.algorithms import TableT
from mipengine.algorithms import TensorT
from mipengine.algorithms import LoopbackTableT
from mipengine.algorithms import LiteralParameterT
from mipengine.algorithms import ScalarT
from mipengine.algorithms import udf
from mipengine.node.udfgen.udfgenerator import UDFGEN_REGISTRY
from mipengine.node.udfgen.udfgenerator import get_generator


PRFX = " " * 4
ANDLN = " AND\n"
SEP = ", "
SEPLN = ",\n"
LN = "\n"
SELECT = "SELECT"
FROM = "FROM"
WHERE = "WHERE"


def all_equal(iterable, func=None):
    """Returns True if all elements in iterable are equal, False otherwise. If
    func is passed a new iterable is used by mapping func to iterable."""
    itr = map(func, iterable) if func else iterable
    g = groupby(itr)
    return next(g, True) and not next(g, False)


def pretify(lst_expr):
    if len(lst_expr) > 80:
        return LN + indent(lst_expr.replace(SEP, SEPLN), PRFX) + LN
    return lst_expr


def generate_udf_select_stmt(
    func_name: str,
    udf_name: str,
    positional_args: list[TableInfo],
    keyword_args: dict[str, TableInfo],  # XXX not used for now
) -> str:
    gen = UDFGEN_REGISTRY.get(func_name, None) or get_generator(func_name)
    parameters = gen.signature.parameters
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
    udf_call_args = pretify(SEP.join(udf_arguments))
    udf_call = f"{udf_name}({udf_call_args})"

    from_subexpr = pretify(SEP.join(table_names))

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

    select_stmt = [SELECT]
    select_stmt.append(indent(udf_call, PRFX))
    select_stmt.append(FROM)
    select_stmt.append(indent(from_subexpr, PRFX))
    select_stmt.append(WHERE)
    select_stmt.append(indent(where_subexpr, PRFX))

    return LN.join(select_stmt)


class ColumnInfo:
    """Mock Columninfo"""

    def __init__(self, name, dtype):
        self.name = name
        self.type = dtype


class TableInfo:
    """Mock TableInfo"""

    def __init__(self, dtype, name, colnames, nrows):
        self.name = name
        self.shape = (nrows, len(colnames))
        self.schema = [ColumnInfo(name, dtype) for name in colnames]


if __name__ == "__main__":
    t1 = TableInfo(int, "tab1", ["a", "b", "c", "d"], 10)
    t2 = TableInfo(int, "tab2", ["A", "B"], 10)
    sel = generate_udf_select_stmt("demo.func", "example", [t1, t2], {})
    print(sel)
    print("-" * 80)
    t1 = TableInfo(int, "tab1", ["dim0", "dim1", "dim2", "val"], 10)
    t2 = TableInfo(int, "tab2", ["dim0", "dim1", "dim2", "val"], 10)
    t3 = TableInfo(int, "tab3", ["dim0", "dim1", "dim2", "val"], 10)
    sel = generate_udf_select_stmt("demo.tensor3", "example", [t1, t2, t3], {})
    print(sel)

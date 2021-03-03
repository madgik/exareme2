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
import mipengine.algorithms
from mipengine.node.udfgen.udfgenerator import UDFGEN_REGISTRY
from mipengine.node.udfgen.udfgenerator import get_generator

TableInfo = TypeVar("TableInfo")


def all_equal(iterable, func=None):
    """Returns True if all elements in iterable are equal, False otherwise. If
    func is passed a new iterable is used by mapping func to iterable."""
    itr = map(func, iterable) if func else iterable
    g = groupby(itr)
    return next(g, True) and not next(g, False)


class Column:
    def __init__(self, name, dtype):
        self.name = name
        self.type = dtype


class Table:
    def __init__(self, dtype, name, colnames, nrows):
        self.name = name
        self.shape = (nrows, len(colnames))
        self.schema = [Column(name, dtype) for name in colnames]


def generate_udf_select_stmt(
    func_name: str,
    udf_name: str,
    positional_args: list[TableInfo],
    keyword_args: dict[str, TableInfo],  # XXX not used for now
) -> str:
    gen = UDFGEN_REGISTRY.get(func_name, None) or get_generator(func_name)
    type_hints = {
        name: param.annotation for name, param in gen.signature.parameters.items()
    }
    if TableT in type_hints.values() and TensorT in type_hints.values():
        #  TODO move this check in udf generator type validation
        raise TypeError("Can't have both TableT and TensorT in udf annotation")

    prfx = " " * 4

    if TableT in type_hints.values():
        tables = [
            arg
            for arg, hint in zip(positional_args, type_hints.values())
            if hint == TableT
        ]
        table_names = [table.name for table in tables]
        table_schemas = [[column.name for column in table.schema] for table in tables]
        udf_arguments = ", ".join(
            [
                f"{table}.{column}"
                for table, columns in zip(table_names, table_schemas)
                for column in columns
            ]
        )
        udf_call = f"{udf_name}({udf_arguments})"
        subselects = []
        for table, columns in zip(table_names, table_schemas):
            col_expr = ", ".join(columns)
            subselects.append(f"(SELECT row_id, {col_expr} FROM {table}) AS {table}")
        subselects = ",\n".join(subselects)
        first_id = f"{table_names[0]}.row_id"
        where_clauses = "\nAND ".join(
            [f"{first_id}={table}.row_id" for table in table_names[1:]]
        )
    elif TensorT in type_hints.values():
        tensors = [
            arg
            for arg, hint in zip(positional_args, type_hints.values())
            if hint == TensorT
        ]
        if not all_equal(tensors, attrgetter("shape")):
            raise TypeError("Can't have tensors of different sizes in python udf")
        ndims = tensors[0].shape[1] - 1
        tensor_names = [tensor.name for tensor in tensors]
        tensor_schemas = [
            [column.name for column in tensor.schema] for tensor in tensors
        ]
        udf_arguments = ", ".join(
            [
                f"{tensor}.{column}"
                for tensor, columns in zip(tensor_names, tensor_schemas)
                for column in columns
            ]
        )
        udf_call = f"{udf_name}({udf_arguments})"
        subselects = []
        for tensor, columns in zip(tensor_names, tensor_schemas):
            col_expr = ", ".join(columns)
            subselects.append(tensor)
        subselects = ",\n".join(subselects)
        all_dims = [f"dim{i}" for i in range(ndims)]
        tensor_dims = [[f"{name}.{dim}" for dim in all_dims] for name in tensor_names]
        first_dims, *rest_dims = tensor_dims
        AND = "\nAND "
        where_clauses = AND.join(
            [
                indent(AND, prfx).join(
                    [f"{a} = {b}" for a, b in zip(first_dims, other)]
                )
                for other in rest_dims
            ]
        )

    select_stmt = ["SELECT"]
    select_stmt.append(indent(udf_call, prfx))
    select_stmt.append("FROM")
    select_stmt.append(indent(subselects, prfx))
    select_stmt.append("WHERE")
    select_stmt.append(indent(where_clauses, prfx))
    return "\n".join(select_stmt)


if __name__ == "__main__":
    t1 = Table(int, "tab1", ["a", "b", "c", "d"], 10)
    t2 = Table(int, "tab2", ["A", "B"], 10)
    sel = generate_udf_select_stmt("demo.func", "example", [t1, t2], {})
    print(sel)
    print("-" * 80)
    t1 = Table(int, "tab1", ["dim0", "dim1", "dim2", "val"], 10)
    t2 = Table(int, "tab2", ["dim0", "dim1", "dim2", "val"], 10)
    t3 = Table(int, "tab3", ["dim0", "dim1", "dim2", "val"], 10)
    sel = generate_udf_select_stmt("demo.tensor3", "example", [t1, t2, t3], {})
    print(sel)

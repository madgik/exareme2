from __future__ import annotations
from textwrap import indent
from typing import TypeVar

import mipengine.algorithms.logistic_regression
from mipengine.node.udfgen.udfgenerator import UDFGEN_REGISTRY
from mipengine.node.udfgen.udfgenerator import get_generator

TableInfo = TypeVar("TableInfo")

TABLE = "TableT"
TENSOR = "TensorT"


class Column:
    def __init__(self, name, dtype):
        self.name = name
        self.type = dtype


class Table:
    def __init__(self, dtype, name, colnames):
        self.name = name
        self.schema = [Column(name, dtype) for name in colnames]


def generate_udf_select_stmt(
    func_name: str,
    udf_name: str,
    positional_args: list[TableInfo],
    keyword_args: dict[str, TableInfo],  # XXX not used for now
) -> str:
    gen = UDFGEN_REGISTRY.get(func_name, None) or get_generator(func_name)
    tables = [
        arg
        for arg, param in zip(positional_args, gen.signature.parameters.values())
        if param.annotation.__name__ in (TABLE, TENSOR)
    ]
    table_names = [table.name for table in tables]
    table_schemas = [[column.name for column in table.schema] for table in tables]
    subselects = []
    for table, columns in zip(table_names, table_schemas):
        col_expr = ",".join(columns)
        subselects.append(
            f"(SELECT ROW_NUMBER() OVER () rowid, {col_expr} FROM {table}) AS {table}"
        )
    subselects = ",\n".join(subselects)
    udf_arguments = ", ".join(
        [
            f"{table}.{column}"
            for table, columns in zip(table_names, table_schemas)
            for column in columns
        ]
    )
    udf_call = f"{udf_name}({udf_arguments})"
    first_id = f"{table_names[0]}.rowid"
    where_clauses = "\nAND ".join(
        [f"{first_id}={table}.rowid" for table in table_names[1:]]
    )

    prfx = " " * 4

    select_stmt = ["SELECT"]
    select_stmt.append(indent(udf_call, prfx))
    select_stmt.append("FROM")
    select_stmt.append(indent(subselects, prfx))
    select_stmt.append("WHERE")
    select_stmt.append(indent(where_clauses, prfx))
    select_stmt.append(";")
    return "\n".join(select_stmt)


if __name__ == "__main__":
    t1 = Table(int, "tab1", ["a", "b"])
    t2 = Table(int, "tab2", ["c", "d"])
    t3 = Table(int, "tab3", ["e", "f"])
    sel = generate_udf_select_stmt(
        "logistic_regression.mat_transp_dot_diag_dot_vec", "example", [t1, t2, t3], {}
    )
    print(sel)

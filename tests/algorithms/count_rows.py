from typing import TypeVar


from mipengine.udfgen import (
    make_unique_func_name,
    relation,
    tensor,
    udf,
)
from mipengine.controller.algorithm_result_DTOs import TabularDataResult
from mipengine.udfgen import merge_tensor
from mipengine.udfgen import scalar


def run(algo_interface):
    local_run = algo_interface.run_udf_on_local_nodes
    global_run = algo_interface.run_udf_on_global_node

    X_relation = algo_interface.initial_view_tables["x"]

    rows = local_run(
        func_name=make_unique_func_name(get_column_rows),
        positional_args={"table": X_relation},
        share_to_global=True,
    )

    sum_of_rows = global_run(
        func_name=make_unique_func_name(get_sum), positional_args={"row_counts": rows}
    )

    total_rows = sum_of_rows.get_table_data()

    result = TabularDataResult(
        title="Rows Count",
        columns=[
            {"name": "rows", "type": "int"},
        ],
        data=[total_rows],
    )
    return result


# ~~~~~~~~~~~~~~~~~~~~~~~~ UDFs ~~~~~~~~~~~~~~~~~~~~~~~~~~ #


S = TypeVar("S")


@udf(table=relation(S), return_type=tensor(int, 1))
def get_column_rows(table):
    rows = [len(table)]
    return rows


@udf(row_counts=merge_tensor(int, 1), return_type=scalar(int))
def get_sum(row_counts):
    total_rows = 0
    for count in row_counts:
        total_rows += count[0]
    return total_rows

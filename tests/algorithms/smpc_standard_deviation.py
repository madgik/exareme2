import json
from typing import TypeVar

from mipengine.table_data_DTOs import ColumnDataFloat
from mipengine.table_data_DTOs import ColumnDataStr
from mipengine.udfgen import (
    make_unique_func_name,
    relation,
    tensor,
    udf,
)
from mipengine.algorithm_result_DTOs import TabularDataResult
from mipengine.udfgen import secure_transfer
from mipengine.udfgen import state
from mipengine.udfgen import transfer


def run(algo_interface):
    local_run = algo_interface.run_udf_on_local_nodes
    global_run = algo_interface.run_udf_on_global_node

    X_relation = algo_interface.initial_view_tables["x"]

    X = local_run(
        func_name=make_unique_func_name(relation_to_matrix),
        positional_args=[X_relation],
    )

    local_state, local_result = local_run(
        func_name=make_unique_func_name(smpc_local_step_1),
        positional_args=[X],
        share_to_global=[False, True],
    )

    global_state, global_result = global_run(
        func_name=make_unique_func_name(smpc_global_step_1),
        positional_args=[local_result],
        share_to_locals=[False, True],
    )

    local_result = local_run(
        func_name=make_unique_func_name(smpc_local_step_2),
        positional_args=[local_state, global_result],
        share_to_global=True,
    )

    global_result = global_run(
        func_name=make_unique_func_name(smpc_global_step_2),
        positional_args=[global_state, local_result],
    )

    std_deviation = json.loads(global_result.get_table_data()[1][0])["deviation"]
    x_variables = algo_interface.x_variables

    result = TabularDataResult(
        title="Standard Deviation",
        columns=[
            ColumnDataStr(name="variable", data=x_variables),
            ColumnDataFloat(name="std_deviation", data=[std_deviation]),
        ],
    )
    return result


# ~~~~~~~~~~~~~~~~~~~~~~~~ UDFs ~~~~~~~~~~~~~~~~~~~~~~~~~~ #


S = TypeVar("S")


@udf(rel=relation(S), return_type=tensor(float, 2))
def relation_to_matrix(rel):
    return rel


@udf(table=tensor(S, 2), return_type=[state(), secure_transfer(add_op=True)])
def smpc_local_step_1(table):
    state_ = {"table": table}
    sum_ = 0
    for (element,) in table:
        sum_ += element
    secure_transfer_ = {
        "sum": {"data": int(sum_), "type": "int", "operation": "addition"},
        "count": {"data": len(table), "type": "int", "operation": "addition"},
    }
    return state_, secure_transfer_


@udf(locals_result=secure_transfer(add_op=True), return_type=[state(), transfer()])
def smpc_global_step_1(locals_result):
    total_sum = locals_result["sum"]
    total_count = locals_result["count"]
    average = total_sum / total_count
    state_ = {"count": total_count}
    transfer_ = {"average": average}
    return state_, transfer_


@udf(
    prev_state=state(),
    global_transfer=transfer(),
    return_type=secure_transfer(add_op=True),
)
def smpc_local_step_2(prev_state, global_transfer):
    deviation_sum = 0
    for (element,) in prev_state["table"]:
        deviation_sum += pow(element - global_transfer["average"], 2)
    secure_transfer_ = {
        "deviation_sum": {
            "data": int(deviation_sum),
            "type": "int",
            "operation": "addition",
        }
    }
    return secure_transfer_


@udf(
    prev_state=state(),
    locals_result=secure_transfer(add_op=True),
    return_type=transfer(),
)
def smpc_global_step_2(prev_state, locals_result):
    total_deviation = locals_result["deviation_sum"]
    from math import sqrt

    deviation = {"deviation": sqrt(total_deviation / prev_state["count"])}
    return deviation

import json
from typing import TypeVar

from mipengine.algorithm_result_DTOs import TabularDataResult
from mipengine.algorithms.algorithm import Algorithm
from mipengine.table_data_DTOs import ColumnDataFloat
from mipengine.table_data_DTOs import ColumnDataStr
from mipengine.udfgen import relation
from mipengine.udfgen import tensor
from mipengine.udfgen import udf
from mipengine.udfgen.udfgenerator import merge_transfer
from mipengine.udfgen.udfgenerator import state
from mipengine.udfgen.udfgenerator import transfer


class StandardDeviationAlgorithm(Algorithm, algname="standard_deviation"):
    def get_variable_groups(self):
        return [self.executor.y_variables]

    def run(self):
        local_run = self.executor.run_udf_on_local_nodes
        global_run = self.executor.run_udf_on_global_node

        Y_relation = self.executor.data_model_views[0]

        Y = local_run(
            func=relation_to_matrix,
            positional_args=[Y_relation],
        )

        local_state, local_result = local_run(
            func=local_step_1,
            positional_args=[Y],
            share_to_global=[False, True],
        )

        global_state, global_result = global_run(
            func=global_step_1,
            positional_args=[local_result],
            share_to_locals=[False, True],
        )

        local_result = local_run(
            func=local_step_2,
            positional_args=[local_state, global_result],
            share_to_global=True,
        )

        global_result = global_run(
            func=global_step_2,
            positional_args=[global_state, local_result],
        )

        std_deviation = json.loads(global_result.get_table_data()[0][0])["deviation"]
        y_variables = self.executor.y_variables

        result = TabularDataResult(
            title="Standard Deviation",
            columns=[
                ColumnDataStr(name="variable", data=y_variables),
                ColumnDataFloat(name="std_deviation", data=[std_deviation]),
            ],
        )
        return result


# ~~~~~~~~~~~~~~~~~~~~~~~~ UDFs ~~~~~~~~~~~~~~~~~~~~~~~~~~ #


S = TypeVar("S")


@udf(rel=relation(S), return_type=tensor(float, 2))
def relation_to_matrix(rel):
    return rel


@udf(table=tensor(S, 2), return_type=[state(), transfer()])
def local_step_1(table):
    state_ = {"table": table}
    sum_ = 0
    for (element,) in table:
        sum_ += element
    transfer_ = {"sum": sum_, "count": len(table)}
    return state_, transfer_


@udf(local_transfers=merge_transfer(), return_type=[state(), transfer()])
def global_step_1(local_transfers):
    total_sum = 0
    total_count = 0
    for transfer in local_transfers:
        total_sum += transfer["sum"]
        total_count += transfer["count"]
    average = total_sum / total_count
    state_ = {"count": total_count}
    transfer_ = {"average": average}
    return state_, transfer_


@udf(prev_state=state(), global_transfer=transfer(), return_type=transfer())
def local_step_2(prev_state, global_transfer):
    deviation_sum = 0
    for (element,) in prev_state["table"]:
        deviation_sum += pow(element - global_transfer["average"], 2)
    transfer_ = {"deviation_sum": deviation_sum}
    return transfer_


@udf(prev_state=state(), local_transfers=merge_transfer(), return_type=transfer())
def global_step_2(prev_state, local_transfers):
    total_deviation_sum = 0
    for transfer in local_transfers:
        total_deviation_sum += transfer["deviation_sum"]
    from math import sqrt

    deviation = {"deviation": sqrt(total_deviation_sum / prev_state["count"])}
    return deviation

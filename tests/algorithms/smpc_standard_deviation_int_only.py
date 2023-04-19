import json
from typing import TypeVar

from mipengine.algorithm_result_DTOs import TabularDataResult
from mipengine.algorithm_specification import AlgorithmSpecification
from mipengine.algorithm_specification import InputDataSpecification
from mipengine.algorithm_specification import InputDataSpecifications
from mipengine.algorithm_specification import InputDataStatType
from mipengine.algorithm_specification import InputDataType
from mipengine.algorithms.algorithm import Algorithm
from mipengine.algorithms.algorithm import AlgorithmDataLoader
from mipengine.table_data_DTOs import ColumnDataFloat
from mipengine.table_data_DTOs import ColumnDataStr
from mipengine.udfgen import relation
from mipengine.udfgen import secure_transfer
from mipengine.udfgen import state
from mipengine.udfgen import tensor
from mipengine.udfgen import transfer
from mipengine.udfgen import udf

ALGORITHM_NAME = "smpc_standard_deviation_int_only"


class StandartDeviationDataLoader(AlgorithmDataLoader, algname=ALGORITHM_NAME):
    def get_variable_groups(self):
        return [self._variables.y]


class StandartDeviationIntOnlySMPC(Algorithm, algname=ALGORITHM_NAME):
    @classmethod
    def get_specification(cls):
        return AlgorithmSpecification(
            name=cls.algname,
            desc="Standard Deviation of a column, transferring only integers, using SMPC",
            label="SMPC Standard Deviation",
            enabled=True,
            inputdata=InputDataSpecifications(
                y=InputDataSpecification(
                    label="column",
                    desc="Column",
                    types=[InputDataType.REAL, InputDataType.INT],
                    stattypes=[InputDataStatType.NUMERICAL],
                    notblank=True,
                    multiple=False,
                )
            ),
        )

    def run(self, data, metadata):
        local_run = self._engine.run_udf_on_local_nodes
        global_run = self._engine.run_udf_on_global_node

        [Y_relation] = data

        Y = local_run(
            func=relation_to_matrix,
            positional_args=[Y_relation],
        )

        local_state, local_result = local_run(
            func=smpc_local_step_1,
            positional_args=[Y],
            share_to_global=[False, True],
        )

        global_state, global_result = global_run(
            func=smpc_global_step_1,
            positional_args=[local_result],
            share_to_locals=[False, True],
        )

        local_result = local_run(
            func=smpc_local_step_2,
            positional_args=[local_state, global_result],
            share_to_global=True,
        )

        global_result = global_run(
            func=smpc_global_step_2,
            positional_args=[global_state, local_result],
        )

        result_data = json.loads(global_result.get_table_data()[0][0])
        std_deviation = result_data["deviation"]
        min_value = result_data["min_value"]
        max_value = result_data["max_value"]
        y_variables = self.variables.y

        result = TabularDataResult(
            title="Standard Deviation",
            columns=[
                ColumnDataStr(name="variable", data=y_variables),
                ColumnDataFloat(name="std_deviation", data=[std_deviation]),
                ColumnDataFloat(name="min_value", data=[min_value]),
                ColumnDataFloat(name="max_value", data=[max_value]),
            ],
        )
        return result


# ~~~~~~~~~~~~~~~~~~~~~~~~ UDFs ~~~~~~~~~~~~~~~~~~~~~~~~~~ #


S = TypeVar("S")


@udf(rel=relation(S), return_type=tensor(float, 2))
def relation_to_matrix(rel):
    return rel


@udf(
    table=tensor(S, 2),
    return_type=[state(), secure_transfer(sum_op=True, min_op=True, max_op=True)],
)
def smpc_local_step_1(table):
    state_ = {"table": table}
    sum_ = 0
    min_value = table[0][0]
    max_value = table[0][0]
    for (element,) in table:
        sum_ += element
        if element < min_value:
            min_value = element
        if element > max_value:
            max_value = element
    secure_transfer_ = {
        "sum": {"data": int(sum_), "operation": "sum", "type": "int"},
        "min": {"data": int(min_value), "operation": "min", "type": "int"},
        "max": {"data": int(max_value), "operation": "max", "type": "int"},
        "count": {"data": len(table), "operation": "sum", "type": "int"},
    }
    return state_, secure_transfer_


@udf(
    locals_result=secure_transfer(sum_op=True, min_op=True, max_op=True),
    return_type=[state(), transfer()],
)
def smpc_global_step_1(locals_result):
    total_sum = locals_result["sum"]
    total_count = locals_result["count"]
    average = total_sum / total_count
    state_ = {
        "count": total_count,
        "min_value": locals_result["min"],
        "max_value": locals_result["max"],
    }
    transfer_ = {"average": average}
    return state_, transfer_


@udf(
    prev_state=state(),
    global_transfer=transfer(),
    return_type=secure_transfer(sum_op=True),
)
def smpc_local_step_2(prev_state, global_transfer):
    deviation_sum = 0
    for (element,) in prev_state["table"]:
        deviation_sum += pow(element - global_transfer["average"], 2)
    secure_transfer_ = {
        "deviation_sum": {
            "data": int(deviation_sum),
            "type": "int",
            "operation": "sum",
        }
    }
    return secure_transfer_


@udf(
    prev_state=state(),
    locals_result=secure_transfer(sum_op=True),
    return_type=transfer(),
)
def smpc_global_step_2(prev_state, locals_result):
    total_deviation = locals_result["deviation_sum"]
    from math import sqrt

    deviation = {
        "deviation": sqrt(total_deviation / prev_state["count"]),
        "min_value": prev_state["min_value"],
        "max_value": prev_state["max_value"],
    }
    return deviation

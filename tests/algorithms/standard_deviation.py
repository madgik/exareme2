import json
from typing import TypeVar

from exareme2.algorithms.exareme2.algorithm import Algorithm
from exareme2.algorithms.exareme2.algorithm import AlgorithmDataLoader
from exareme2.algorithms.exareme2.udfgen import merge_transfer
from exareme2.algorithms.exareme2.udfgen import relation
from exareme2.algorithms.exareme2.udfgen import state
from exareme2.algorithms.exareme2.udfgen import tensor
from exareme2.algorithms.exareme2.udfgen import transfer
from exareme2.algorithms.exareme2.udfgen import udf
from exareme2.algorithms.specifications import AlgorithmSpecification
from exareme2.algorithms.specifications import InputDataSpecification
from exareme2.algorithms.specifications import InputDataSpecifications
from exareme2.algorithms.specifications import InputDataStatType
from exareme2.algorithms.specifications import InputDataType
from exareme2.worker_communication import ColumnDataFloat
from exareme2.worker_communication import ColumnDataStr
from exareme2.worker_communication import TabularDataResult

ALGORITHM_NAME = "standard_deviation"


class StandardDeviationDataLoader(AlgorithmDataLoader, algname=ALGORITHM_NAME):
    def get_variable_groups(self):
        return [self._variables.y]


class StandardDeviationAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    @classmethod
    def get_specification(cls):
        return AlgorithmSpecification(
            name=cls.algname,
            desc="Standard Deviation of a column",
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
        local_run = self.engine.run_udf_on_local_workers
        global_run = self.engine.run_udf_on_global_worker

        [Y_relation] = data

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
        y_variables = self.variables.y

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

from typing import TypeVar

from pandas import DataFrame

from mipengine.udfgen import relation
from mipengine.udfgen import scalar
from mipengine.udfgen import udf
from mipengine.udfgen.udfgenerator import secure_transfer
from mipengine.udfgen.udfgenerator import state
from mipengine.udfgen.udfgenerator import transfer

S = TypeVar("S")


@udf(table=relation(S), return_type=scalar(int))
def get_column_rows(table):
    rows = [len(table)]
    return rows


@udf(table=relation(S), return_type=[state(), transfer()])
def local_step(table: DataFrame):
    sum_ = 0
    for element, *_ in table.values:
        sum_ += element
    sum_ = int(sum_)
    transfer_ = {"sum": sum_, "count": len(table)}
    state_ = {"sum": sum_, "count": len(table)}
    return state_, transfer_


@udf(table=relation(S), return_type=secure_transfer(sum_op=True))
def smpc_local_step(table: DataFrame):
    sum_ = 0
    for element, *_ in table.values:
        sum_ += element
    secure_transfer_ = {"sum": {"data": int(sum_), "operation": "sum"}}
    return secure_transfer_


@udf(locals_result=secure_transfer(sum_op=True), return_type=transfer())
def smpc_global_step(locals_result):
    result = {"total_sum": locals_result["sum"]}
    return result


@udf(table=relation(S), return_type=scalar(int))
def very_slow_udf(table):
    from time import sleep

    sleep(1000)
    rows = [len(table)]
    return rows

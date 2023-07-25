from typing import TypeVar

from pandas import DataFrame

from exareme2.udfgen import relation
from exareme2.udfgen import secure_transfer
from exareme2.udfgen import state
from exareme2.udfgen import transfer
from exareme2.udfgen import udf

S = TypeVar("S")


@udf(table=relation(S), return_type=relation([("res", int)]))
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
    secure_transfer_ = {"sum": {"data": int(sum_), "operation": "sum", "type": "int"}}
    return secure_transfer_


@udf(locals_result=secure_transfer(sum_op=True), return_type=transfer())
def smpc_global_step(locals_result):
    result = {"total_sum": locals_result["sum"]}
    return result


@udf(table=relation(S), return_type=relation([("res", int)]))
def one_second_udf(table):
    from time import sleep

    sleep(1)
    rows = [len(table)]
    return rows


@udf(table=relation(S), return_type=relation([("res", int)]))
def five_seconds_udf(table):
    from time import sleep

    sleep(5)
    rows = [len(table)]
    return rows


@udf(table=relation(S), return_type=relation([("res", int)]))
def one_hundred_seconds_udf(table):
    from time import sleep

    sleep(100)
    rows = [len(table)]
    return rows

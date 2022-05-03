import numpy
import pandas as pd

from typing import TypeVar

from mipengine.algorithm_result_DTOs import TabularDataResult
from mipengine.table_data_DTOs import ColumnDataFloat
from mipengine.table_data_DTOs import ColumnDataStr
from mipengine.udfgen import TensorBinaryOp
from mipengine.udfgen import TensorUnaryOp
from mipengine.udfgen import literal
from mipengine.udfgen import merge_tensor
from mipengine.udfgen import relation
from mipengine.udfgen import tensor
from mipengine.udfgen import udf

from mipengine.udfgen import transfer
from mipengine.udfgen import merge_transfer
from mipengine.udfgen import state

import json

T = TypeVar('T')
S = TypeVar("S")

def run(algo_interface):
    local_run = algo_interface.run_udf_on_local_nodes
    global_run = algo_interface.run_udf_on_global_node

    X_relation = algo_interface.initial_view_tables["x"]

    X = local_run(
        func=relation_to_matrix,
        positional_args=[X_relation],
    )

    local_result = local_run(
        func=local_stats,
        positional_args=[X],
        share_to_global=[True],
    )

    global_state,global_result = global_run(
        func=global_stats,
        positional_args=[local_result],
        share_to_locals=[False,True]
    )

    #print(global_result.get_table_data())
    #print("printing table data 0 "+str(global_result.get_table_data()[0][0]))
    #print("printing table data 1 "+str(global_result.get_table_data()[1][0]))

    new_max = json.loads(global_result.get_table_data()[1][0])["max"]
    new_min = json.loads(global_result.get_table_data()[1][0])["min"]
    new_mean = json.loads(global_result.get_table_data()[1][0])["mean"]
    count_not_null = json.loads(global_result.get_table_data()[1][0])["count_not_null"]
    count_num_null = json.loads(global_result.get_table_data()[1][0])["count_num_null"]
    #print(new_max)
    new_quartiles = json.loads(global_result.get_table_data()[1][0])["quartiles"]

    #print(new_quartiles)
    quartiles_array = numpy.array(new_quartiles)
    #print(quartiles_array.shape)
    q1_array = quartiles_array[:,0,:]
    q2_array = quartiles_array[:,1,:]
    q3_array = quartiles_array[:,2,:]
    #print(q1_array)

    q1_final = numpy.apply_along_axis(str,0, q1_array).tolist()
    q2_final = numpy.apply_along_axis(str,0, q2_array).tolist()
    q3_final = numpy.apply_along_axis(str,0, q3_array).tolist()

    x_variables = algo_interface.x_variables



    #up to here it works

    local_result2 = local_run(
        func=local_step_2,
        positional_args=[X,global_result],
        share_to_global=True,
    )

    global_result2 = global_run(
        func=global_step_2,
        positional_args=[global_state,local_result2],
        share_to_locals=[False]
    )

    new_std = json.loads(global_result2.get_table_data()[1][0])["deviation"]


    #return result
    #up to here it works
    X_not_null = local_run(
        func=remove_nulls,
        positional_args=[X],
    )

    local_result_not_null = local_run(
        func=local_stats,
        positional_args=[X_not_null],
        share_to_global=[True],
    )

    global_state_nn,global_result_not_null = global_run(
        func=global_stats,
        positional_args=[local_result_not_null],
        share_to_locals=[False,True]
    )

    new_max_nn = json.loads(global_result_not_null.get_table_data()[1][0])["max"]
    new_min_nn = json.loads(global_result_not_null.get_table_data()[1][0])["min"]
    new_mean_nn = json.loads(global_result_not_null.get_table_data()[1][0])["mean"]
    count_not_null_nn = json.loads(global_result_not_null.get_table_data()[1][0])["count_not_null"]
    count_num_null_nn = json.loads(global_result_not_null.get_table_data()[1][0])["count_num_null"]
    new_quartiles_nn = json.loads(global_result_not_null.get_table_data()[1][0])["quartiles"]

    quartiles_array_nn = numpy.array(new_quartiles_nn)
    #print(quartiles_array.shape)
    q1_array_nn = quartiles_array_nn[:,0,:]
    q2_array_nn = quartiles_array_nn[:,1,:]
    q3_array_nn = quartiles_array_nn[:,2,:]
    #print(q1_array)

    q1_final_nn = numpy.apply_along_axis(str,0, q1_array_nn).tolist()
    q2_final_nn = numpy.apply_along_axis(str,0, q2_array_nn).tolist()
    q3_final_nn = numpy.apply_along_axis(str,0, q3_array_nn).tolist()

    local_result2_nn = local_run(
        func=local_step_2,
        positional_args=[X_not_null,global_result_not_null],
        share_to_global=[True],
    )

    global_result2_nn = global_run(
        func=global_step_2,
        positional_args=[global_state_nn,local_result2_nn],
    )

    new_std_nn = json.loads(global_result2_nn.get_table_data()[1][0])["deviation"]

    result = TabularDataResult(
        title="STATS",
        columns=[
            ColumnDataStr(name="variable", data=x_variables),
            ColumnDataFloat(name="max", data=new_max),
            ColumnDataFloat(name="min", data=new_min),
            ColumnDataFloat(name="mean", data=new_mean),
            ColumnDataFloat(name="count_not_null", data=count_not_null),
            ColumnDataFloat(name="count_num_null", data=count_num_null),
            ColumnDataFloat(name="std", data=new_std),
            ColumnDataStr(name="q1", data=q1_final),
            ColumnDataStr(name="q2", data=q2_final),
            ColumnDataStr(name="q3", data=q3_final),
            ColumnDataFloat(name="max_model", data=new_max_nn),
            ColumnDataFloat(name="min_model", data=new_min_nn),
            ColumnDataFloat(name="mean_model", data=new_mean_nn),
            ColumnDataFloat(name="count_not_null_model", data=count_not_null_nn),
            ColumnDataFloat(name="count_num_null_model", data=count_num_null_nn),
            ColumnDataFloat(name="std_model", data=new_std_nn),
            ColumnDataStr(name="q1_model", data=q1_final_nn),
            ColumnDataStr(name="q2_model", data=q2_final_nn),
            ColumnDataStr(name="q3_model", data=q3_final_nn),
            ])
    return result


@udf(rel=relation(S), return_type=tensor(float, 2))
def relation_to_matrix(rel):
    return rel

@udf(a=tensor(T, 2), return_type=tensor(T, 2))
def remove_nulls(a):
    a_sel = a[~numpy.isnan(a).any(axis=1)]
    return a_sel


@udf(input_array=tensor(dtype=T, ndims=2), return_type=[transfer()])
def local_stats(input_array):
    local_max = numpy.nanmax(input_array,axis=0)
    local_min = numpy.nanmin(input_array,axis=0)
    local_sum = numpy.nansum(input_array,axis=0)
    quartiles = numpy.nanpercentile(input_array, [25, 50, 75],axis =0)
    num_null = numpy.isnan(input_array).sum(axis= 0)
    local_count = input_array.shape[0]
    local_count_not_null = input_array.shape[0] - num_null

    transfer = {}
    transfer['max']=local_max.tolist()
    transfer['min']=local_min.tolist()
    transfer['sum']=local_sum.tolist()
    transfer['count']=local_count
    transfer['count_not_null'] = local_count_not_null.tolist()
    transfer['num_null'] = num_null.tolist()
    transfer['quartiles'] = quartiles.tolist()

    return transfer

@udf(local_transfers=merge_transfer(), return_type=[state(),transfer()])
def global_stats(local_transfers):

    curr_max = local_transfers[0]["max"]
    curr_min = local_transfers[0]["min"]
    ncols = len(curr_min)
    curr_sum = numpy.zeros(ncols)
    curr_count = numpy.zeros(ncols)
    curr_count_not_null = numpy.zeros(ncols)
    curr_count_null = numpy.zeros(ncols)
    quartiles_list = []

    for curr_transfer in local_transfers:
        curr_max = numpy.maximum(curr_max,curr_transfer["max"])
        curr_min = numpy.minimum(curr_max,curr_transfer["min"])
        curr_sum += curr_transfer['sum']
        curr_count += curr_transfer['count']
        curr_count_not_null += curr_transfer['count_not_null']
        curr_count_null += curr_transfer['num_null']
        quartiles_list.append(curr_transfer['quartiles'])

    total_mean = curr_sum/curr_count_not_null
    state_ = {}
    final_result = {'max':curr_max.tolist(),'min':curr_min.tolist(),
                 'mean':total_mean.tolist(),'count_not_null':curr_count_not_null.tolist(),
                 'count_num_null': curr_count_null.tolist(),'quartiles':quartiles_list}
    state_['count_not_null'] = curr_count_not_null.tolist()
    return state_,final_result

@udf(input_array=tensor(dtype=T, ndims=2), global_transfer=transfer(), return_type=transfer())
def local_step_2(input_array, global_transfer):
    #deviation_sum = 0
    #for (element,) in prev_state["table"]:
        #deviation_sum += pow(element - global_transfer["average"], 2)
    curr_mean = global_transfer["mean"]
    deviation_sum = numpy.nansum((input_array - curr_mean)**2,axis=0)

    transfer_ = {"deviation_sum": deviation_sum.tolist()}
    #raise ValueError(str(deviation_sum) +' '+str(curr_mean))
    return transfer_


@udf(prev_state=state(), local_transfers=merge_transfer(), return_type=transfer())
def global_step_2(prev_state, local_transfers):
    ncols = len(local_transfers[0]['deviation_sum'])
    total_deviation_sum = numpy.zeros(ncols)
    for curr_transfer in local_transfers:
        total_deviation_sum += curr_transfer["deviation_sum"]

    count_not_null = prev_state["count_not_null"]
    temp_array = total_deviation_sum / count_not_null
    final_deviation = numpy.sqrt(temp_array)

    #raise ValueError('tds: '+str(total_deviation_sum)+)

    deviation = {"deviation": final_deviation.tolist()}
    return deviation

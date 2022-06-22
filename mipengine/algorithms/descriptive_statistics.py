import numpy
import pandas as pd

from typing import TypeVar
from typing import Dict

from typing import List

from pydantic import BaseModel

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

class DescriptiveResult(BaseModel):
    title: str
    variable: List[str]
    numerical_variables: List[str]
    categorical_variables: List[str]
    categorical_counts: List[Dict]
    max: List[float]
    min: List[float]
    mean: List[float]
    count_not_null: List[float]
    count_num_null: List[float]
    std: List[float]
    q1: List[List[float]]
    q2: List[List[float]]
    q3: List[List[float]]
    max_model: List[float]
    min_model: List[float]
    mean_model: List[float]
    count_not_null_model: List[float]
    count_num_null_model: List[float]
    std_model: List[float]
    q1_model: List[List[float]]
    q2_model: List[List[float]]
    q3_model: List[List[float]]


def run(algo_interface):
    local_run = algo_interface.run_udf_on_local_nodes
    global_run = algo_interface.run_udf_on_global_node

    #X_relation = algo_interface.initial_view_tables["x"]
    x_list = algo_interface.x_variables if algo_interface.x_variables != None else []
    y_list = algo_interface.y_variables if algo_interface.y_variables != None else []

    X_relation, *_ = algo_interface.create_primary_data_views(
        #variable_groups=[x_list+y_list],dropna=False
        variable_groups=[x_list+y_list],dropna=False
    )

    metadata_dict = algo_interface.metadata

    categorical_columns = []
    numerical_columns = []
    for colname,rest_dict in metadata_dict.items():
        if rest_dict['is_categorical'] == True:
            categorical_columns.append(colname)
        else:
            numerical_columns.append(colname)
    print(categorical_columns)
    print(numerical_columns)

    #categorical_relation = local_run(
    #    func=filter_categorical,
    #    positional_args=[X_relation,categorical_columns],
    #    share_to_global=[True],
    #)


    local_result_categorical = local_run(
        func=count_locals,
        positional_args=[X_relation,categorical_columns],
        share_to_global=[True],
    )

    global_categorical = global_run(
        func=global_counts,
        positional_args=[local_result_categorical,categorical_columns],
        share_to_locals=[False]
    )

    global_categorical_res = json.loads(global_categorical.get_table_data()[1][0])
    categorical_counts_list = []
    for curr_column in categorical_columns:
        categorical_counts_list.append(global_categorical_res[curr_column])
    """
    X_numeric = local_run(
        func=filter_numerical,
        positional_args=[X_relation,numerical_columns],
    )
    X = local_run(
        func=relation_to_matrix,
        positional_args=[X_numeric],
    )
    """

    X = local_run(
        func=relation_to_matrix_num,
        positional_args=[X_relation,numerical_columns],
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

    q1_final = q1_array.tolist()
    q2_final = q2_array.tolist()
    q3_final = q3_array.tolist()

    x_variables = x_list



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

    q1_final_nn = q1_array_nn.tolist()
    q2_final_nn = q2_array_nn.tolist()
    q3_final_nn = q3_array_nn.tolist()

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

    result = DescriptiveResult(
        title="Descriptive Statistics",
        variable= x_variables,
        numerical_variables= numerical_columns,
        categorical_variables= categorical_columns,
        categorical_counts = categorical_counts_list,
        max=new_max,
        min=new_min,
        mean=new_mean,
        count_not_null= count_not_null,
        count_num_null= count_num_null,
        std=new_std,
        q1= q1_final,
        q2= q2_final,
        q3= q3_final,
        max_model=new_max_nn,
        min_model= new_min_nn,
        mean_model= new_mean_nn,
        count_not_null_model=count_not_null_nn,
        count_num_null_model=count_num_null_nn,
        std_model=new_std_nn,
        q1_model=q1_final_nn,
        q2_model=q2_final_nn,
        q3_model=q3_final_nn
    )
    return result

@udf(input_df=relation(S),columns_list =literal(),return_type=[transfer()])
def count_locals(input_df,columns_list):
    values_list1 = []
    #raise(ValueError(input_df.head(5)))
    for curr_column in columns_list:
        filled_values1 = input_df['input_df_'+curr_column].fillna("NaN")
        values_list1.append(filled_values1.value_counts(dropna=False).to_dict())
    ret_dict1 = {}
    for curr_column,curr_dict in zip(columns_list,values_list1):
        ret_dict1[curr_column] = curr_dict
    return ret_dict1

@udf(local_transfers=merge_transfer(),columns_list =literal(), return_type=[transfer()])
def global_counts(local_transfers,columns_list):
    from collections import Counter

    final_dict = {}
    for curr_column in columns_list:
        first_dict = local_transfers[0][curr_column]
        res = Counter(first_dict)
        rem_transfers = local_transfers[1:]
        for curr_transfer in rem_transfers:
            c = curr_transfer[curr_column]
            res += Counter(c)
        final_dict[curr_column] = dict(res)
    return final_dict

@udf(rel=relation(S), return_type=tensor(float, 2))
def relation_to_matrix(rel):
    return rel

@udf(input_df=relation(S),categorical_columns=literal(), return_type=relation(S))
def filter_categorical(input_df,categorical_columns):
    raise ValueError(input_df.head(5))
    ret_categorical = ['input_df_'+ curr_cat for curr_cat in categorical_columns]
    ret_rel = input_df[ret_categorical]
    return ret_rel

@udf(input_df=relation(S),numerical_columns=literal(), return_type=relation(S))
def filter_numerical(input_df,numerical_columns):
    raise ValueError(input_df.head(5))
    ret_numerical = ['input_df_'+ curr_num for curr_num in numerical_columns]
    ret_rel = input_df[ret_numerical]
    return ret_rel

@udf(rel=relation(S),numerical_columns=literal(), return_type=tensor(float, 2))
def relation_to_matrix_num(rel,numerical_columns):
    ret_numerical = ['rel_'+ curr_num for curr_num in numerical_columns]
    ret_rel = rel[ret_numerical]
    return ret_rel

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

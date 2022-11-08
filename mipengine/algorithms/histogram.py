from typing import List
from typing import TypeVar

import numpy
from pydantic import BaseModel

from mipengine.algorithms.helpers import get_transfer_data
from mipengine.udfgen import literal
from mipengine.udfgen import merge_transfer
from mipengine.udfgen import relation
from mipengine.udfgen import secure_transfer
from mipengine.udfgen import state
from mipengine.udfgen import transfer
from mipengine.udfgen import udf

S = TypeVar("S")


class HistogramResult(BaseModel):
    numerical: dict
    categorical: dict


def run(algo_interface):
    local_run = algo_interface.run_udf_on_local_nodes
    global_run = algo_interface.run_udf_on_global_node

    xvars = algo_interface.x_variables or []
    yvars = algo_interface.y_variables or []

    # bins = 20
    bins = algo_interface.algorithm_parameters["bins"]

    [data] = algo_interface.create_primary_data_views(
        variable_groups=[algo_interface.y_variables + algo_interface.x_variables],
    )

    metadata = dict(algo_interface.metadata)
    # print(metadata)
    vars = [var for var in xvars + yvars if var != "dataset"]

    numerical_vars = [var for var in vars if not metadata[var]["is_categorical"]]
    nominal_vars = [var for var in vars if metadata[var]["is_categorical"]]

    enumerations_dict = {var: metadata[var]["enumerations"] for var in nominal_vars}
    print(enumerations_dict)

    yvar = yvars[0]
    print(yvar)
    print(xvars)
    if yvar in nominal_vars:
        locals_result = local_run(
            func=compute_local_histogram_categorical_secure,
            positional_args=[data, enumerations_dict, yvar, xvars],
            share_to_global=[True],
        )

        categorical_histogram = get_transfer_data(
            global_run(
                func=merge_grouped_histogram_categorical_secure2,
                positional_args=[locals_result, xvars],
                share_to_locals=[False],
            )
        )
        numerical_histogram = {}

    else:
        # find_min_max_local()
        local_result4 = local_run(
            func=find_min_max_local,
            positional_args=[data, yvar],
            share_to_global=[True],
        )
        # find_min_max_global()
        min_max_result = get_transfer_data(
            global_run(
                func=find_min_max_global,
                positional_args=[local_result4],
                share_to_locals=[False],
            )
        )
        min_value = min_max_result["min_value"]
        max_value = min_max_result["max_value"]
        # compute_local_histogram_numerical(data,bins,yvar,xvars,min_value,max_value)
        local_result5 = local_run(
            func=compute_local_histogram_numerical_secure2,
            positional_args=[
                data,
                enumerations_dict,
                bins,
                yvar,
                xvars,
                min_value,
                max_value,
            ],
            share_to_global=[True],
        )
        # merge_local_histogram_numerical(local_transfers)

        numerical_histogram = get_transfer_data(
            global_run(
                func=merge_grouped_histogram_numerical_secure2,
                positional_args=[local_result5, xvars],
                share_to_locals=[False],
            )
        )
        categorical_histogram = {}

    ret_val = HistogramResult(
        numerical=numerical_histogram, categorical=categorical_histogram
    )

    return ret_val


@udf(
    data=relation(S),
    column=literal(),
    return_type=[secure_transfer(sum_op=True, min_op=True, max_op=True)],
)
def find_min_max_local(data, column):
    min_value = data[column].min()
    max_value = data[column].max()

    secure_transfer_ = {
        "min": {"data": float(min_value), "operation": "min", "type": "float"},
        "max": {"data": float(max_value), "operation": "max", "type": "float"},
    }

    return secure_transfer_


@udf(
    locals_result=secure_transfer(sum_op=True, min_op=True, max_op=True),
    return_type=[transfer()],
)
def find_min_max_global(locals_result):

    transfer_ = {
        "min_value": locals_result["min"],
        "max_value": locals_result["max"],
    }

    return transfer_


@udf(
    data=relation(S),
    metadata=literal(),
    yvar=literal(),
    xvars=literal(),
    return_type=[secure_transfer(sum_op=True, min_op=True, max_op=True)],
)
def compute_local_histogram_categorical_secure(data, metadata, yvar, xvars):
    possible_enumerations = metadata[yvar].keys()
    local_counts = data[yvar].value_counts().to_dict()
    categorical_histogram_list = []
    for curr_key in sorted(possible_enumerations):
        curr_value = local_counts.get(curr_key, 0)
        categorical_histogram_list.append(curr_value)

    secure_transfer_ = {}
    secure_transfer_["categorical_histogram"] = {
        "data": categorical_histogram_list,
        "operation": "sum",
        "type": "int",
    }
    if xvars:
        final_dict = {}
        for x_variable in xvars:
            local_grouped_histogram = {}
            grouped = data[[yvar, x_variable]].groupby(x_variable)
            for group_name, curr_grouped in grouped:
                local_grouped_histogram[group_name] = (
                    curr_grouped[yvar].value_counts().to_dict()
                )
            final_dict[x_variable] = local_grouped_histogram

            possible_groups = sorted(metadata[x_variable].keys())

            for curr_group in possible_groups:
                curr_result = final_dict[x_variable].get(curr_group, {})
                final_dict[x_variable][curr_group] = curr_result

        grouped_list = []
        for x_variable in xvars:
            possible_groups = sorted(metadata[x_variable].keys())
            possible_values = sorted(metadata[yvar].keys())
            groups_list = []
            for curr_group in possible_groups:
                elements_list = []
                for curr_element in possible_values:
                    curr_result = final_dict[x_variable][curr_group].get(
                        curr_element, 0
                    )
                    elements_list.append(curr_result)
                groups_list.append(elements_list)
            grouped_list.append(groups_list)
        secure_transfer_["grouped_histogram_categorical"] = {
            "data": grouped_list,
            "operation": "sum",
            "type": "int",
        }
    return secure_transfer_


@udf(
    locals_result=secure_transfer(sum_op=True, min_op=True, max_op=True),
    xvars=literal(),
    return_type=[transfer()],
)
def merge_grouped_histogram_categorical_secure2(locals_result, xvars):
    return_dict = {}
    return_dict["categorical_histogram"] = locals_result["categorical_histogram"]
    if xvars:
        return_dict["grouped_histogram_categorical"] = locals_result[
            "grouped_histogram_categorical"
        ]
    return return_dict


@udf(
    data=relation(S),
    metadata=literal(),
    bins=literal(),
    yvar=literal(),
    xvars=literal(),
    min_value=literal(),
    max_value=literal(),
    return_type=[secure_transfer(sum_op=True, min_op=True, max_op=True)],
)
def compute_local_histogram_numerical_secure2(
    data, metadata, bins, yvar, xvars, min_value, max_value
):
    # yvar = algo_interface.y_variables[0]
    def hist_func(x, min_value, max_value, bins=20):
        hist, bins = numpy.histogram(x, range=(min_value, max_value), bins=bins)
        return hist

    final_dict = {}
    local_histogram = hist_func(
        data[yvar], min_value=min_value, max_value=max_value, bins=bins
    )
    secure_transfer_ = {}
    secure_transfer_["histogram"] = {
        "data": local_histogram.tolist(),
        "operation": "sum",
        "type": "int",
    }
    if xvars:
        final_dict = {}
        for x_variable in xvars:
            local_grouped_histogram = (
                data[[yvar, x_variable]]
                .groupby(x_variable)
                .apply(
                    lambda x: hist_func(
                        x, min_value=min_value, max_value=max_value, bins=bins
                    ).tolist()
                )
                .to_dict()
            )
            final_dict[x_variable] = local_grouped_histogram
        # transfer_={}
        # transfer_['grouped_histogram']= final_dict

        for x_variable in xvars:
            for curr_group in sorted(metadata[x_variable].keys()):
                result = final_dict[x_variable].get(
                    curr_group, numpy.zeros(bins, dtype="int64").tolist()
                )
                final_dict[x_variable][curr_group] = result
        x_variables_list = []
        for x_variable in xvars:
            groups_list = []
            for curr_group in sorted(metadata[x_variable].keys()):
                curr_element = final_dict[x_variable][curr_group]
                groups_list.append(curr_element)
            x_variables_list.append(groups_list)

        secure_transfer_["grouped_histogram"] = {
            "data": x_variables_list,
            "operation": "sum",
            "type": "int",
        }

    return secure_transfer_


@udf(
    locals_result=secure_transfer(sum_op=True, min_op=True, max_op=True),
    xvars=literal(),
    return_type=[transfer()],
)
def merge_grouped_histogram_numerical_secure2(locals_result, xvars):
    return_dict = {}
    return_dict["histogram"] = locals_result["histogram"]
    if xvars:
        return_dict["grouped_histogram"] = locals_result["grouped_histogram"]
    return return_dict

from collections import Counter
from typing import List
from typing import TypeVar

import numpy
from pydantic import BaseModel

from mipengine.algorithms.helpers import get_transfer_data
from mipengine.udfgen import relation
from mipengine.udfgen import secure_transfer
from mipengine.udfgen import state
from mipengine.udfgen import transfer
from mipengine.udfgen import udf


def run(algo_interface):
    local_run = algo_interface.run_udf_on_local_nodes
    global_run = algo_interface.run_udf_on_global_node

    xvars = algo_interface.x_variables or []
    yvars = algo_interface.y_variables or []

    bins = 20

    [data] = algo_interface.create_primary_data_views(
        variable_groups=[algo_interface.y_variables + algo_interface.x_variables],
    )

    metadata = algo_interface.metadata

    vars = [var for var in xvars + yvars if var != "dataset"]

    numerical_vars = [var for var in vars if not metadata[var]["is_categorical"]]
    nominal_vars = [var for var in vars if metadata[var]["is_categorical"]]

    if yvars[0] in nominal_vars:
        local_result = local_run(
            func=compute_local_histogram_categorical,
            positional_args=[data, yvars],
            share_to_global=[True],
        )

        categorical_histogram = get_transfer_data(
            global_run(
                func=merge_local_histogram_categorical,
                positional_args=[local_result, yvars[0]],
                share_to_locals=[False],
            )
        )
        # compute_local_histogram_categorical()
        # merge_local_histogram_categorical()

        if xvars is not []:
            local_result2 = local_run(
                func=compute_groups_local,
                positional_args=[data, xvars, yvars[0]],
                share_to_global=[True],
            )
            # compute_groups_local()
            groups = get_transfer_data(
                global_run(
                    func=compute_groups_global,
                    positional_args=[local_result2, xvars],
                    share_to_locals=[True],
                )
            )

            # compute_groups_global()
            # compute_local_grouped_categorical()
            local_result3 = local_run(
                func=compute_local_grouped_categorical,
                positional_args=[data, yvars, xvars],
                share_to_global=[True],
            )
            # merge_local_grouped_categorical()
            histogram_categorical_grouped = get_transfer_data(
                global_run(
                    func=merge_grouped_categorical,
                    positional_args=[local_result3, xvars, groups],
                    share_to_locals=[False],
                )
            )
    else:
        # find_min_max_local()
        local_result4 = local_run(
            func=find_min_max_local,
            positional_args=[data, yvars[0]],
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
            func=compute_local_histogram_numerical,
            positional_args=[data, bins, yvars[0], xvars, min_value, max_value],
            share_to_global=[True],
        )
        # merge_local_histogram_numerical(local_transfers)

        numerical_histogram = get_transfer_data(
            global_run(
                func=merge_local_histogram_numerical,
                positional_args=[local_result5],
                share_to_locals=[False],
            )
        )

        if xvars is not None:
            # compute_groups_local()
            # compute_groups_global()
            local_result6 = local_run(
                func=compute_groups_local,
                positional_args=[data, xvars, yvars[0]],
                share_to_global=[True],
            )
            # compute_groups_local()
            groups = get_transfer_data(
                global_run(
                    func=compute_groups_global,
                    positional_args=[local_result6, xvars],
                    share_to_locals=[True],
                )
            )

            # compute_local_grouped_numerical(data,yvar,xvars,min_value,max_value,bins)
            local_result7 = local_run(
                func=compute_local_grouped_numerical,
                positional_args=[data, yvars[0], xvars, min_value, max_value, bins],
                share_to_global=[True],
            )
            # merge_grouped_histogram_numerical(local_transfers,groups,xvars,bins)

            grouped_histogram_numerical = get_transfer_data(
                global_run(
                    func=merge_grouped_histogram_numerical,
                    positional_args=[local_result7, groups, xvars, bins],
                    share_to_locals=[False],
                )
            )


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

    return transfer


@udf(
    data=relation(S),
    xvars=literal(),
    y_var=literal(),
    return_type=[transfer()],
)
def compute_groups_local(data, xvars, y_var):
    groups_list = []
    curr_dict = {}
    for xvar in sorted(xvars):
        grouped = data[[y_var, xvar]].groupby(xvar)
        curr_dict[xvar] = list(grouped.groups.keys())
    return curr_dict


@udf(
    local_transfers=merge_transfer(),
    xvars=literal(),
    return_type=[transfer()],
)
def compute_groups_global(local_transfers, xvars):
    final_groups = {}
    for xvar in xvars:
        target_list = []
        for curr_transfer in local_transfers:
            target_list += curr_transfer[xvar]
        curr_groups = list(set(target_list))
        final_groups[xvar] = curr_groups
    return final_groups


@udf(
    data=relation(S),
    yvars=literal(),
    return_type=[transfer()],
)
def compute_local_histogram_categorical(data, yvars):

    # values = Y_relation.value_counts()
    counts = {var: data[var].value_counts().to_dict() for var in yvars}
    return counts


@udf(
    local_transfers=merge_transfer(),
    yvar=literal(),
    return_type=[transfer()],
)
def merge_local_histogram_categorical(local_transfers, yvar):
    result = Counter({})
    for curr_transfer in local_transfers:
        result += Counter(curr_transfer[yvar])
    final_dict = dict(result)
    return final_dict


def hist_func(x, min_value, max_value, bins=20):
    hist, bins = numpy.histogram(x, range=(min_value, max_value), bins=bins)
    return hist


@udf(
    data=relation(S),
    bins=literal(),
    yvar=literal(),
    xvars=literal(),
    min_value=literal(),
    max_value=literal(),
    return_type=[transfer()],
)
def compute_local_histogram_numerical(data, bins, yvar, xvars, min_value, max_value):
    # yvar = algo_interface.y_variables[0]
    final_dict = {}
    local_histogram = hist_func(
        data[yvar], min_value=min_value, max_value=max_value, bins=bins
    )
    transfer_ = {}
    transfer_["histogram"] = local_histogram.tolist()
    return transfer_


@udf(
    data=relation(S),
    bins=literal(),
    yvar=literal(),
    xvars=literal(),
    min_value=literal(),
    max_value=literal(),
    return_type=[secure_transfer(sum_op=True, min_op=True, max_op=True)],
)
def compute_local_histogram_numerical_secure(
    data, bins, yvar, xvars, min_value, max_value
):
    # yvar = algo_interface.y_variables[0]
    final_dict = {}
    local_histogram = hist_func(
        data[yvar], min_value=min_value, max_value=max_value, bins=bins
    )

    secure_transfer_ = {
        "histogram": {
            "data": local_histogram.tolist(),
            "operation": "sum",
            "type": "list",
        },
    }


@udf(
    locals_result=secure_transfer(sum_op=True, min_op=True, max_op=True),
    return_type=[transfer()],
)
def merge_local_histogram_numerical_secure(locals_result):
    result = {}
    result["histogram"] = locals_result["histogram"]
    return result


@udf(
    local_transfers=merge_transfer(),
    return_type=[transfer()],
)
def merge_local_histogram_numerical(local_transfers):
    result = {}
    curr_result = numpy.array(local_transfers[0]["histogram"])
    for curr_transfer in local_transfers[1:]:
        curr_result += numpy.array(curr_transfer["histogram"])
    result["merged_histogram"] = curr_result.tolist()
    return result


@udf(
    data=relation(S),
    yvar=literal(),
    xvars=literal(),
    min_value=literal(),
    max_value=literal(),
    bins=literal(),
    return_type=[transfer()],
)
def compute_local_grouped_numerical(data, yvar, xvars, min_value, max_value, bins):
    # yvar = algo_interface.y_variables[0]
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
    transfer_ = {}
    transfer_["grouped_histogram"] = final_dict
    return transfer_


@udf(
    data=relation(S),
    yvar=literal(),
    xvars=literal(),
    min_value=literal(),
    max_value=literal(),
    bins=literal(),
    groups=literal(),
    return_type=[secure_transfer(sum_op=True, min_op=True, max_op=True)],
)
def compute_local_grouped_numerical_secure(
    data, yvar, xvars, min_value, max_value, bins, groups
):
    # yvar = algo_interface.y_variables[0]
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
        for curr_group in groups[x_variable]:
            result = final_dict[x_variable].get(
                curr_group, numpy.zeros(bins, dtype="int64").tolist()
            )
            final_dict[x_variable][curr_group] = result
    x_variables_list = []
    for x_variable in xvars:
        groups_list = []
        for curr_group in groups[x_variable]:
            curr_element = final_dict[x_variable][curr_group]
            groups_list.append(curr_element)
        x_variables_list.append(groups_list)

    secure_transfer_ = {
        "grouped_histogram": {
            "data": x_variables_list,
            "operation": "sum",
            "type": "list",
        },
    }
    return secure_transfer_


@udf(
    locals_result=secure_transfer(sum_op=True, min_op=True, max_op=True),
    return_type=[transfer()],
)
def merge_grouped_histogram_numerical_secure(locals_result):

    return locals_result["grouped_histogram"]


@udf(
    local_transfers=merge_transfer(),
    groups=literal(),
    xvars=literal(),
    bins=literal(),
    return_type=[transfer()],
)
def merge_grouped_histogram_numerical(local_transfers, groups, xvars, bins):
    final_dict = {}
    for x_variable in xvars:
        final_dict[x_variable] = {}
        for curr_group in groups[x_variable]:
            curr_sum = numpy.array(
                local_transfers[0]["grouped_histogram"][x_variable].get(
                    curr_group, numpy.zeros(bins, dtype="int64").tolist()
                )
            )
            for curr_transfer in local_transfers[1:]:
                curr_local = numpy.array(
                    curr_transfer["grouped_histogram"][x_variable].get(
                        curr_group, numpy.zeros(bins, dtype="int64").tolist()
                    )
                )
                curr_sum += curr_local
            final_dict[x_variable][curr_group] = curr_sum.tolist()
    return final_dict


@udf(
    data=relation(S),
    yvar=literal(),
    xvars=literal(),
    return_type=[transfer()],
)
def compute_local_grouped_categorical(data, yvar, xvars):
    # yvar = algo_interface.y_variables[0]
    final_dict = {}
    for x_variable in xvars:
        local_grouped_histogram = {}
        grouped = data[[yvar, x_variable]].groupby(x_variable)
        for group_name, curr_grouped in grouped:
            local_grouped_histogram[group_name] = (
                curr_grouped[yvar].value_counts().to_dict()
            )
        final_dict[x_variable] = local_grouped_histogram
    transfer_ = {}
    transfer_["grouped_histogram"] = final_dict
    return transfer_


@udf(
    local_transfers=merge_transfer(),
    xvars=literal(),
    groups=literal(),
    return_type=[transfer()],
)
def merge_grouped_categorical(local_transfers, xvars, groups):
    # yvar = algo_interface.y_variables[0]
    final_dict = {}
    for x_variable in xvars:
        final_dict[x_variable] = {}
        for curr_group in groups[x_variable]:
            # print(curr_group)
            final_dict[x_variable][curr_group] = {}
            curr_result = Counter(
                local_transfers[0]["grouped_histogram"][x_variable].get(curr_group, {})
            )
            # print(curr_result)
            for curr_transfer in local_transfers[1:]:
                curr_result += Counter(
                    curr_transfer["grouped_histogram"][x_variable].get(curr_group, {})
                )
            final_dict[x_variable][curr_group] = dict(curr_result)
    return dict(final_dict)


@udf(
    data=relation(S),
    yvar=literal(),
    xvars=literal(),
    return_type=[transfer()],
)
def compute_values_local(data, yvar, xvars):
    final_dict = {}
    # print(yvar)
    # print(xvars)
    for x_variable in xvars:
        final_dict[x_variable] = {}
        grouped = data[[yvar, x_variable]].groupby(x_variable)
        for group_name, curr_grouped in grouped:
            # print(group_name)
            final_dict[x_variable][group_name] = {}
            final_dict[x_variable][group_name] = list(
                curr_grouped[yvar].value_counts().to_dict().keys()
            )
    return final_dict


@udf(
    local_transfers=merge_transfer(),
    xvars=literal(),
    groups=literal(),
    return_type=[transfer()],
)
def merge_values_categorical(local_transfers, xvars, groups):
    merge_dict = {}
    for x_variable in xvars:
        merge_dict[x_variable] = {}
        groups2 = groups[x_variable]
        for curr_group in groups2:
            merge_dict[x_variable][curr_group] = {}
            curr_result = local_transfers[0][x_variable]
            curr_result2 = curr_result.get(curr_group, [])
            for curr_transfer in local_transfers[1:]:
                curr_result3 = curr_transfer[x_variable]
                curr_result4 = curr_result3.get(curr_group, [])
                curr_result2 += curr_result4
            merge_dict[x_variable][curr_group] = curr_result2

    return merge_dict


@udf(
    data=relation(S),
    groups=literal(),
    merge_dict=literal(),
    yvar=literal(),
    xvars=literal(),
    return_type=[secure_transfer(sum_op=True, min_op=True, max_op=True)],
)
def compute_local_histogram_categorical_secure(data, groups, merge_dict, yvar, xvars):
    def value_dictionaries_to_list(groups, merge_dict, local_dict, xvars):
        x_variables_list = []
        for x_variable in xvars:
            groups2 = groups[x_variable]
            groups_list = []
            for curr_group in groups2:
                elements_list = []
                for curr_element in merge_dict[x_variable][curr_group]:
                    curr_count = local_dict[x_variable][curr_group].get(curr_element, 0)
                    elements_list.append(curr_count)
                groups_list.append(elements_list)
            x_variables_list.append(groups_list)
        return x_variables_list

    final_dict = {}
    for x_variable in xvars:
        local_grouped_histogram = {}
        grouped = data[[yvar, x_variable]].groupby(x_variable)
        for group_name, curr_grouped in grouped:
            local_grouped_histogram[group_name] = (
                curr_grouped[yvar].value_counts().to_dict()
            )
        final_dict[x_variable] = local_grouped_histogram

    for x_variable in xvars:
        groups2 = groups[x_variable]
        for curr_group in groups2:
            curr_result = final_dict[x_variable].get(curr_group, {})
            final_dict[x_variable][curr_group] = curr_result
    local_list = value_dictionaries_to_list(groups, merge_dict, final_dict, xvars)

    secure_transfer_ = {
        "grouped_histogram_categorical": {
            "data": local_list,
            "operation": "sum",
            "type": "list",
        },
    }
    return secure_transfer_


@udf(
    locals_result=secure_transfer(sum_op=True, min_op=True, max_op=True),
    return_type=[transfer()],
)
def merge_grouped_histogram_categorical_secure(locals_result):

    return locals_result["grouped_histogram_categorical"]

import json
from typing import TypeVar

import pandas as pd
from pydantic import BaseModel

from mipengine.algorithms.helpers import get_transfer_data
from mipengine.udfgen import secure_transfer
from mipengine.udfgen.udfgenerator import literal
from mipengine.udfgen.udfgenerator import merge_transfer
from mipengine.udfgen.udfgenerator import relation
from mipengine.udfgen.udfgenerator import transfer
from mipengine.udfgen.udfgenerator import udf


class AnovaResult(BaseModel):
    n_obs: float
    grand_mean: float
    sum_sq_x1: float
    sum_sq_x2: float


S = TypeVar("S")


def run(algo_interface):
    local_run = algo_interface.run_udf_on_local_nodes
    global_run = algo_interface.run_udf_on_global_node

    xvars, yvars = algo_interface.x_variables, algo_interface.y_variables
    X, y = algo_interface.create_primary_data_views([xvars, yvars])
    covar_enums_x1 = list(algo_interface.metadata[xvars[0]]["enumerations"])
    covar_enums_x2 = list(algo_interface.metadata[xvars[1]]["enumerations"])
    sstype = algo_interface.algorithm_parameters["sstype"]

    sec_local_transfer, local_transfers = local_run(
        func=local1,
        keyword_args=dict(
            y=y,
            x=X,
            covar_enums_x1=covar_enums_x1,
            covar_enums_x2=covar_enums_x2,
            sstype=sstype,
        ),
        share_to_global=[True, True],
    )

    result = global_run(
        func=global1,
        keyword_args=dict(
            sec_local_transfer=sec_local_transfer, local_transfers=local_transfers
        ),
    )

    result = get_transfer_data(result)

    anova_table = AnovaResult(
        n_obs=result["n_obs"],
        grand_mean=result["grand_mean"],
        sum_sq_x1=result["sum_sq_x1"],
        sum_sq_x2=result["sum_sq_x2"],
    )

    return anova_table


T = TypeVar("T")


@udf(
    y=relation(schema=S),
    x=relation(schema=T),
    covar_enums_x1=literal(),
    covar_enums_x2=literal(),
    sstype=literal(),
    return_type=[secure_transfer(sum_op=True, min_op=True, max_op=True), transfer()],
)
def local1(y, x, covar_enums_x1, covar_enums_x2, sstype):
    import numpy as np
    import pandas as pd

    n_obs = y.shape[0]
    variable = y.reset_index(drop=True).to_numpy().squeeze()
    var_label = y.columns.values.tolist()[0]
    covar_label_x1 = x.columns[0]
    covar_label_x2 = x.columns[1]
    covar_x1 = x[covar_label_x1].values.tolist()
    covar_x2 = x[covar_label_x2].values.tolist()
    covar_enums_x1 = np.array(covar_enums_x1)
    covar_enums_x2 = np.array(covar_enums_x2)

    # Create dataframe
    dataset = pd.DataFrame()
    dataset[var_label] = variable
    dataset[covar_label_x1] = covar_x1
    dataset[covar_label_x2] = covar_x2
    var_sq = var_label + "_sq"
    dataset[var_sq] = variable**2

    # Calculate degrees of freedom
    dof_x1 = len(covar_enums_x1) - 1
    dof_x2 = len(covar_enums_x2) - 1
    dof_x1x2 = dof_x1 * dof_x2
    dof_w = n_obs - (len(covar_enums_x1) * len(covar_enums_x2))

    sum_y = sum(y[var_label])
    sum_y_sqrd = sum(y[var_label] ** 2)

    # get overall stats
    overall_stats = dataset[var_label].agg(["count", "sum"])
    overall_ssq = dataset[var_sq].sum()
    overall_stats = overall_stats.append(pd.Series(data=overall_ssq, index=["sum_sq"]))

    group_stats_x1 = (
        dataset[[var_label, covar_label_x1]]
        .groupby(covar_label_x1)
        .agg(["count", "sum"])
    )
    group_stats_x1.columns = ["count", "sum"]
    group_ssq_x1 = dataset[[var_sq, covar_label_x1]].groupby(covar_label_x1).sum()
    group_ssq_x1.columns = ["sum_sq"]
    group_stats_x1["group_ssq_x1"] = group_ssq_x1

    group_stats_x2 = (
        dataset[[var_label, covar_label_x2]]
        .groupby(covar_label_x2)
        .agg(["count", "sum"])
    )
    group_stats_x2.columns = ["count", "sum"]
    group_ssq_x2 = dataset[[var_sq, covar_label_x2]].groupby(covar_label_x2).sum()
    group_ssq_x2.columns = ["sum_sq"]
    group_stats_x2["group_ssq_x2"] = group_ssq_x2

    group_stats_df_y_x1 = pd.DataFrame(group_stats_x1)
    group_stats_df_y_x2 = pd.DataFrame(group_stats_x2)
    group_by_x1_x2 = dataset.groupby([covar_label_x1, covar_label_x2]).agg(
        ["count", "sum"]
    )

    # Find differences between enumerations and current index of data to find discrepancies
    diff_x1 = list(set(covar_enums_x1) - set(group_stats_x1.index))
    diff_x2 = list(set(covar_enums_x2) - set(group_stats_x2.index))

    if diff_x1:
        diff_df_y_x1 = pd.DataFrame(
            0,
            index=diff_x1,
            columns=["count", "sum", "group_ssq"],
        )
        group_stats_df_y_x1 = group_stats_x1.append(diff_df_y_x1)
    if diff_x2:
        diff_df_y_x2 = pd.DataFrame(
            0,
            index=diff_x2,
            columns=["count", "sum", "group_ssq"],
        )
        group_stats_df_y_x2 = diff_df_y_x2.append(group_stats_x2)

    group_stats_df_y_x1["groups"] = pd.Categorical(
        group_stats_df_y_x1.index, categories=covar_enums_x1.tolist(), ordered=True
    )
    group_stats_df_y_x1.sort_values("groups", inplace=True)

    group_stats_df_y_x2["groups"] = pd.Categorical(
        group_stats_df_y_x2.index, categories=covar_enums_x2.tolist(), ordered=True
    )
    group_stats_df_y_x2.sort_values("groups", inplace=True)

    sec_transfer_ = {}
    sec_transfer_["n_obs"] = {"data": n_obs, "operation": "sum", "type": "int"}
    sec_transfer_["sum_y"] = {
        "data": sum_y,
        "operation": "sum",
        "type": "float",
    }
    sec_transfer_["sum_y_sqrd"] = {
        "data": sum_y_sqrd,
        "operation": "sum",
        "type": "float",
    }
    sec_transfer_["overall_stats_sum"] = {
        "data": overall_stats["sum"].tolist(),
        "operation": "sum",
        "type": "float",
    }
    sec_transfer_["overall_stats_count"] = {
        "data": overall_stats["count"].tolist(),
        "operation": "sum",
        "type": "float",
    }
    sec_transfer_["overall_ssq"] = {
        "data": overall_ssq.item(),
        "operation": "sum",
        "type": "float",
    }
    sec_transfer_["group_stats_x1_sum"] = {
        "data": group_stats_df_y_x1["sum"].tolist(),
        "operation": "sum",
        "type": "float",
    }
    sec_transfer_["group_stats_x1_count"] = {
        "data": group_stats_df_y_x1["count"].tolist(),
        "operation": "sum",
        "type": "float",
    }
    sec_transfer_["group_stats_x1_ssq"] = {
        "data": group_stats_df_y_x1["group_ssq_x1"].tolist(),
        "operation": "sum",
        "type": "float",
    }
    sec_transfer_["group_stats_x2_sum"] = {
        "data": group_stats_df_y_x2["sum"].tolist(),
        "operation": "sum",
        "type": "float",
    }
    sec_transfer_["group_stats_x2_count"] = {
        "data": group_stats_df_y_x2["count"].tolist(),
        "operation": "sum",
        "type": "float",
    }
    sec_transfer_["group_stats_x2_ssq"] = {
        "data": group_stats_df_y_x2["group_ssq_x2"].tolist(),
        "operation": "sum",
        "type": "float",
    }
    sec_transfer_["group_by_x1_x2_sum"] = {
        "data": group_by_x1_x2[var_label]["sum"].tolist(),
        "operation": "sum",
        "type": "float",
    }
    sec_transfer_["group_by_x1_x2_count"] = {
        "data": group_by_x1_x2[var_label]["count"].tolist(),
        "operation": "sum",
        "type": "float",
    }
    sec_transfer_["group_by_x1_x2_index"] = {
        "data": group_by_x1_x2[var_label].index.tolist(),
        "operation": "sum",
        "type": "float",
    }

    transfer_ = {
        "var_label": var_label,
        "dof_w": dof_w,
        "dof_x1": dof_x1,
        "dof_x2": dof_x2,
        "dof_x1x2": dof_x1x2,
        "covar_label_x1": covar_label_x1,
        "covar_label_x2": covar_label_x2,
        "covar_enums_x1": covar_enums_x1.tolist(),
        "covar_enums_x2": covar_enums_x2.tolist(),
        "group_stats_df_y_x1_index": group_stats_df_y_x1.index.tolist(),
        "group_stats_df_y_x2_index": group_stats_df_y_x2.index.tolist(),
    }

    return sec_transfer_, transfer_


@udf(
    sec_local_transfer=secure_transfer(sum_op=True, min_op=True, max_op=True),
    local_transfers=merge_transfer(),
    return_type=[transfer()],
)
def global1(sec_local_transfer, local_transfers):
    import itertools

    import numpy as np

    n_obs = sec_local_transfer["n_obs"]
    sum_y = sec_local_transfer["sum_y"]
    group_stats_x1_sum = np.array(sec_local_transfer["group_stats_x1_sum"])
    group_stats_x1_count = np.array(sec_local_transfer["group_stats_x1_count"])
    group_ssq_x1 = np.array(sec_local_transfer["group_stats_x1_ssq"])
    group_stats_x2_sum = np.array(sec_local_transfer["group_stats_x2_sum"])
    group_stats_x2_count = np.array(sec_local_transfer["group_stats_x2_count"])
    group_ssq_x2 = np.array(sec_local_transfer["group_stats_x2_ssq"])

    group_stats_df_y_x1_index = [
        t["group_stats_df_y_x1_index"] for t in local_transfers
    ]
    group_stats_df_y_x2_index = [
        t["group_stats_df_y_x2_index"] for t in local_transfers
    ]

    group_stats_index_x1 = group_stats_df_y_x1_index[0]
    if len(group_stats_df_y_x1_index) > 1:
        for x, y in itertools.combinations(group_stats_df_y_x1_index, 2):
            group_stats_index_x1 = x
            diff = list(set(x) - set(y))
            if diff != []:
                group_stats_index_x1.append(diff)

    group_stats_index_x2 = group_stats_df_y_x2_index[0]
    if len(group_stats_df_y_x2_index) > 1:
        for x, y in itertools.combinations(group_stats_df_y_x2_index, 2):
            group_stats_index_x2 = x
            diff = list(set(x) - set(y))
            if diff != []:
                group_stats_index_x2.append(diff)

    # remove zero count groups
    df_y_x1 = pd.DataFrame(
        {
            "groups": group_stats_index_x1,
            "count": group_stats_x1_count,
            "ssq": group_ssq_x1,
            "sum": group_stats_x1_sum,
        },
        index=group_stats_index_x1,
    )
    if not np.all(group_stats_x1_count):
        df_y_x1 = df_y_x1[df_y_x1["count"] != 0]
        group_stats_x1_count = np.array(df_y_x1["count"])
        group_stats_x1_sum = np.array(df_y_x1["sum"])

    # remove zero count groups
    df_y_x2 = pd.DataFrame(
        {
            "groups": group_stats_index_x2,
            "count": group_stats_x2_count,
            "ssq": group_ssq_x2,
            "sum": group_stats_x2_sum,
        },
        index=group_stats_index_x2,
    )
    if not np.all(group_stats_x2_count):
        df_y_x2 = df_y_x2[df_y_x2["count"] != 0]
        group_stats_x2_count = np.array(df_y_x2["count"])
        group_stats_x2_sum = np.array(df_y_x2["sum"])

    categories_x1 = local_transfers[0]["covar_enums_x1"]
    if len(categories_x1) < 2:
        raise ValueError("Cannot perform Anova when there is only one level")

    categories_x2 = local_transfers[0]["covar_enums_x2"]
    if len(categories_x2) < 2:
        raise ValueError("Cannot perform Anova when there is only one level")

    grand_mean = sum_y / n_obs
    df_y_x1["mean"] = group_stats_x1_sum / group_stats_x1_count
    df_y_x2["mean"] = group_stats_x2_sum / group_stats_x2_count
    ssq_x1 = sum((df_y_x1["mean"] - grand_mean) ** 2)
    ssq_x2 = sum((df_y_x2["mean"] - grand_mean) ** 2)

    transfer_ = {
        "n_obs": n_obs,
        "grand_mean": grand_mean,
        "sum_sq_x1": ssq_x1,
        "sum_sq_x2": ssq_x2,
    }

    return transfer_

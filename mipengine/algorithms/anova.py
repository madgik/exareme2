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
    [data] = algo_interface.create_primary_data_views([yvars + xvars])
    covar_enums_x1 = list(algo_interface.metadata[xvars[0]]["enumerations"])
    covar_enums_x2 = list(algo_interface.metadata[xvars[1]]["enumerations"])
    sstype = algo_interface.algorithm_parameters["sstype"]

    if len(covar_enums_x1) < 2:
        raise ValueError("Cannot perform Anova when there is only one level")

    if len(covar_enums_x2) < 2:
        raise ValueError("Cannot perform Anova when there is only one level")

    sec_local_transfer, local_transfers = local_run(
        func=local1,
        keyword_args=dict(
            data=data,
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
    data=relation(schema=T),
    covar_enums_x1=literal(),
    covar_enums_x2=literal(),
    sstype=literal(),
    return_type=[secure_transfer(sum_op=True, min_op=True, max_op=True), transfer()],
)
def local1(data, covar_enums_x1, covar_enums_x2, sstype):
    import numpy as np
    import pandas as pd

    n_obs = data.shape[0]
    var_label = data.columns[0]
    covar_label_x1 = data.columns[1]
    covar_label_x2 = data.columns[2]
    covar_enums_x1 = np.array(covar_enums_x1)
    covar_enums_x2 = np.array(covar_enums_x2)

    data["var_sq"] = data[var_label] ** 2
    group_by_x1_x2 = data.groupby([covar_label_x1, covar_label_x2]).agg(
        ["count", "sum"]
    )

    sec_transfer_ = {}
    sec_transfer_["y_per_x1_x2_group_count"] = {
        "data": group_by_x1_x2[var_label]["count"].tolist(),
        "operation": "sum",
        "type": "float",
    }
    sec_transfer_["y_per_x1_x2_group_sum"] = {
        "data": group_by_x1_x2[var_label]["sum"].tolist(),
        "operation": "sum",
        "type": "float",
    }
    sec_transfer_["y_per_x1_x2_group_ssq"] = {
        "data": group_by_x1_x2["var_sq"]["sum"].tolist(),
        "operation": "sum",
        "type": "float",
    }
    sec_transfer_["table_index"] = {
        "data": group_by_x1_x2.index.tolist(),
        "operation": "sum",
        "type": "float",
    }

    transfer_ = {
        "var_label": var_label,
        "covar_label_x1": covar_label_x1,
        "covar_label_x2": covar_label_x2,
        "covar_enums_x1": covar_enums_x1.tolist(),
        "covar_enums_x2": covar_enums_x2.tolist(),
    }

    return sec_transfer_, transfer_


@udf(
    sec_local_transfer=secure_transfer(sum_op=True, min_op=True, max_op=True),
    local_transfers=merge_transfer(),
    return_type=[transfer()],
)
def global1(sec_local_transfer, local_transfers):
    import numpy as np
    import pandas as pd

    group_stats_sum = np.array(sec_local_transfer["y_per_x1_x2_group_sum"])
    group_stats_count = np.array(sec_local_transfer["y_per_x1_x2_group_count"])
    group_stats_ssq = np.array(sec_local_transfer["y_per_x1_x2_group_ssq"])
    group_table_index = np.array(sec_local_transfer["table_index"])
    covar_enums_x1 = local_transfers[0]["covar_enums_x1"]
    covar_enums_x2 = local_transfers[0]["covar_enums_x2"]
    var_label = [t["var_label"] for t in local_transfers][0]
    covar_label_x1 = [t["covar_label_x1"] for t in local_transfers][0]
    covar_label_x2 = [t["covar_label_x2"] for t in local_transfers][0]

    res_dict = {
        "x1_index": group_table_index.T[0],
        "x2_index": group_table_index.T[1],
        "n": group_stats_count,
        "sx": group_stats_sum,
        "sxx": group_stats_ssq,
    }

    df = pd.DataFrame(res_dict)
    df.set_index(["x1_index"])

    n = df.pivot(index="x1_index", columns="x2_index", values="n")
    sx = df.pivot(index="x1_index", columns="x2_index", values="sx")
    sxx = df.pivot(index="x1_index", columns="x2_index", values="sxx")

    nh_denom = sum(1 / df["n"])
    n_harmonic = (len(covar_enums_x1) * len(covar_enums_x2)) / nh_denom
    ss_cell = sxx - (sx**2) / n

    cell_means = sx / n
    total_means_x1 = cell_means.sum(axis=1)
    total_means_x2 = cell_means.sum(axis=0)
    all_means = total_means_x1.tolist() + total_means_x2.tolist()
    all_means_sq = [j**2 for j in all_means]
    s_x1_sq = sum(total_means_x1**2) / len(covar_enums_x2)
    s_x2_sq = sum(total_means_x2**2) / len(covar_enums_x1)
    s_x1x2_sq = sum(all_means_sq)
    n_obs = sum(df["n"])
    sum_y = sum(df["sx"])
    sum_y_ssq = sum(df["sxx"])
    grand_mean = sum(total_means_x1) ** 2 / (len(covar_enums_x1) * len(covar_enums_x2))
    ssq_x1 = n_harmonic * (s_x1_sq - grand_mean)
    ssq_x2 = n_harmonic * (s_x2_sq - grand_mean)
    ssq_x1x2 = n_harmonic * (s_x1x2_sq - s_x1_sq - s_x2_sq + grand_mean)
    # raise ValueError(ssq_x1, ssq_x2, s_x1x2_sq)

    # Calculate degrees of freedom
    dof_x1 = len(covar_enums_x1) - 1
    dof_x2 = len(covar_enums_x2) - 1
    dof_x1x2 = dof_x1 * dof_x2
    dof_w = n_obs - (len(covar_enums_x1) * len(covar_enums_x2))

    transfer_ = {
        "n_obs": n_obs,
        "grand_mean": grand_mean,
        "sum_sq_x1": ssq_x1,
        "sum_sq_x2": ssq_x2,
    }

    return transfer_

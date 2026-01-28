from typing import Dict
from typing import List

import numpy as np
import pandas as pd
from pydantic import BaseModel

from exaflow.algorithms.exareme3.algorithm import Algorithm
from exaflow.algorithms.exareme3.exareme3_registry import exareme3_udf
from exaflow.algorithms.exareme3.library.anova_common import get_min_max_ci_info
from exaflow.worker_communication import BadUserInput

ALGORITHM_NAME = "anova_oneway"


class AnovaResult(BaseModel):
    anova_table: Dict
    tuckey_test: List[Dict]
    min_max_per_group: Dict
    ci_info: Dict


class AnovaOneWayAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self, metadata: dict):
        """
        Exaflow implementation of one-way ANOVA with Tukey HSD, matching the
        behavior of the original exaflow ANOVA_ONEWAY algorithm.
        """
        y_var_name = self.inputdata.y[0]
        x_var_name = self.inputdata.x[0]
        covar_enums = metadata[x_var_name].get("enumerations")

        # Run a single distributed ANOVA UDF
        udf_results = self.engine.run_algorithm_udf(
            func=anova_oneway_local_step,
            positional_args={
                "x_var": x_var_name,
                "y_var": y_var_name,
                "covar_enums": covar_enums,
            },
        )

        result = udf_results[0]

        anova_result = {
            "n_obs": result["n_obs"],
            "y_label": y_var_name,
            "x_label": x_var_name,
            "df_residual": result["df_residual"],
            "df_explained": result["df_explained"],
            "ss_residual": result["ss_residual"],
            "ss_explained": result["ss_explained"],
            "ms_residual": result["ms_residual"],
            "ms_explained": result["ms_explained"],
            "p_value": result["p_value"],
            "f_stat": result["f_stat"],
        }

        # Tukey table into list-of-dicts (records)
        tukey_result = {
            "groupA": result["thsd A"],
            "groupB": result["thsd B"],
            "meanA": result["thsd mean(A)"],
            "meanB": result["thsd mean(B)"],
            "diff": result["thsd diff"],
            "se": result["thsd Std.Err."],
            "t_stat": result["thsd t value"],
            "p_tuckey": result["thsd Pr(>|t|)"],
        }
        df_tukey_result = pd.DataFrame(tukey_result)
        tukey_test = df_tukey_result.to_dict(orient="records")

        means = result["means"]
        sample_stds = result["sample_stds"]
        categories = result["categories"]
        var_min_per_group = result["var_min_per_group"]
        var_max_per_group = result["var_max_per_group"]
        group_stats_index = result["group_stats_index"]

        min_max_per_group, ci_info = get_min_max_ci_info(
            means=means,
            sample_stds=sample_stds,
            categories=categories,
            var_min_per_group=var_min_per_group,
            var_max_per_group=var_max_per_group,
            group_stats_index=group_stats_index,
        )

        return AnovaResult(
            anova_table=anova_result,
            tuckey_test=tukey_test,
            min_max_per_group=min_max_per_group,
            ci_info=ci_info,
        )


# ---------------------------------------------------------------------------
# UDF: local computation + secure aggregation
# ---------------------------------------------------------------------------


@exareme3_udf(with_aggregation_server=True)
def anova_oneway_local_step(data, agg_client, x_var, y_var, covar_enums):
    """
    Exaflow UDF that:
    - On each worker: builds local group statistics for y by x.
    - Uses agg_client to aggregate sums / counts / ssq / min / max securely.
    - On the aggregation server: computes ANOVA + Tukey HSD from global stats.
    """
    import itertools
    import sys

    import scipy.stats as st
    from statsmodels.stats.libqsturng import psturng

    covar_enums = list(covar_enums)

    # --- Local stats like original local1, but force 1D arrays ---
    y_col = data[y_var]
    x_col = data[x_var]

    # If for any reason we got a DataFrame, fall back to the first column
    if isinstance(y_col, pd.DataFrame):
        y_col = y_col.iloc[:, 0]
    if isinstance(x_col, pd.DataFrame):
        x_col = x_col.iloc[:, 0]

    y_col = y_col.reset_index(drop=True)
    x_col = x_col.reset_index(drop=True)

    variable = y_col.to_numpy().reshape(-1)  # 1D array
    covariable = x_col.to_numpy().reshape(-1)  # 1D array
    covar_enums_arr = np.array(covar_enums)

    dataset = pd.DataFrame({y_var: variable, x_var: covariable}, copy=False)
    n_obs_local = dataset.shape[0]

    min_per_group_local = dataset.groupby([x_var]).min()
    max_per_group_local = dataset.groupby([x_var]).max()

    var_sq = f"{y_var}_sq"
    dataset[var_sq] = variable**2

    # overall stats
    overall_stats = dataset[y_var].agg(["count", "sum"])
    overall_ssq = dataset[var_sq].sum()

    # group stats
    group_stats = dataset[[y_var, x_var]].groupby(x_var).agg(["count", "sum"])
    group_stats.columns = ["count", "sum"]
    group_ssq_local = dataset[[var_sq, x_var]].groupby(x_var).sum()
    group_ssq_local.columns = ["sum_sq"]

    min_per_group_local.columns = ["min_per_group"]
    max_per_group_local.columns = ["max_per_group"]

    group_stats_df = pd.DataFrame(group_stats)
    group_stats_df["group_ssq"] = group_ssq_local
    group_stats_df["min_per_group"] = min_per_group_local
    group_stats_df["max_per_group"] = max_per_group_local

    # Add rows for categories not present locally
    diff = list(set(covar_enums_arr) - set(group_stats.index))
    if diff:
        diff_df = pd.DataFrame(
            0,
            index=diff,
            columns=["count", "sum", "group_ssq", "min_per_group", "max_per_group"],
        )
        diff_df["min_per_group"] = sys.float_info.max
        diff_df["max_per_group"] = sys.float_info.min
        group_stats_df = pd.concat([group_stats_df, diff_df])

    group_stats_df["groups"] = pd.Categorical(
        group_stats_df.index, categories=covar_enums_arr.tolist(), ordered=True
    )
    group_stats_df.sort_values("groups", inplace=True)

    # local arrays in the common (aligned) order
    group_stats_sum_local = group_stats_df["sum"].to_numpy(dtype=float)
    group_stats_count_local = group_stats_df["count"].to_numpy(dtype=float)
    group_ssq_local = group_stats_df["group_ssq"].to_numpy(dtype=float)
    var_min_per_group_local = group_stats_df["min_per_group"].to_numpy(dtype=float)
    var_max_per_group_local = group_stats_df["max_per_group"].to_numpy(dtype=float)

    # --- Secure aggregation across workers (batch to avoid mode mixing) ---
    total_n_obs_arr = agg_client.sum(np.array([float(n_obs_local)], dtype=float))
    total_overall_sum_arr = agg_client.sum(
        np.array([float(overall_stats["sum"])], dtype=float)
    )
    total_overall_count_arr = agg_client.sum(
        np.array([float(overall_stats["count"])], dtype=float)
    )
    total_overall_ssq_arr = agg_client.sum(np.array([float(overall_ssq)], dtype=float))
    group_stats_sum_arr = agg_client.sum(group_stats_sum_local)
    group_stats_count_arr = agg_client.sum(group_stats_count_local)
    group_ssq_arr = agg_client.sum(group_ssq_local)
    var_min_arr = agg_client.min(var_min_per_group_local)
    var_max_arr = agg_client.max(var_max_per_group_local)

    n_obs = int(np.asarray(total_n_obs_arr).reshape(-1)[0])
    overall_stats_sum = float(np.asarray(total_overall_sum_arr).reshape(-1)[0])
    overall_stats_count = float(np.asarray(total_overall_count_arr).reshape(-1)[0])
    overall_ssq = float(np.asarray(total_overall_ssq_arr).reshape(-1)[0])
    group_stats_sum = np.asarray(group_stats_sum_arr, dtype=float)
    group_stats_count = np.asarray(group_stats_count_arr, dtype=float)
    group_ssq = np.asarray(group_ssq_arr, dtype=float)
    var_min_per_group = np.asarray(var_min_arr, dtype=float)
    var_max_per_group = np.asarray(var_max_arr, dtype=float)

    # At this point, we are effectively on the aggregation server,
    # with global stats in group_stats_* arrays.

    # Remove groups with zero count
    nonzero_mask = group_stats_count != 0
    if not np.all(nonzero_mask):
        group_stats_count = group_stats_count[nonzero_mask]
        group_stats_sum = group_stats_sum[nonzero_mask]
        group_ssq = group_ssq[nonzero_mask]
        var_min_per_group = var_min_per_group[nonzero_mask]
        var_max_per_group = var_max_per_group[nonzero_mask]
        group_stats_index = [c for c, m in zip(covar_enums, nonzero_mask) if m]
    else:
        group_stats_index = list(covar_enums)

    categories = covar_enums
    if len(categories) < 2 or len(group_stats_index) < 2:
        raise BadUserInput(
            "Cannot perform Anova one-way. Covariable has only one level."
        )

    # Degrees of freedom
    df_explained = len(group_stats_index) - 1
    df_residual = n_obs - len(group_stats_index)

    # Sums of squares
    overall_mean = (
        overall_stats_sum / overall_stats_count if overall_stats_count else 0.0
    )
    ss_residual = overall_ssq - np.sum(group_stats_sum**2 / group_stats_count)
    ss_explained = np.sum(
        (overall_mean - group_stats_sum / group_stats_count) ** 2 * group_stats_count
    )

    ms_explained = ss_explained / df_explained
    ms_residual = ss_residual / df_residual
    f_stat = ms_explained / ms_residual if ms_residual != 0 else 0.0
    p_value = 1.0 - st.f.cdf(f_stat, df_explained, df_residual)

    # ANOVA summary table (internal)
    anova_table = {
        "df_explained": df_explained,
        "df_residual": df_residual,
        "ss_explained": ss_explained,
        "ss_residual": ss_residual,
        "ms_explained": ms_explained,
        "ms_residual": ms_residual,
        "f_stat": f_stat,
        "p_value": p_value,
    }

    # Tukey HSD
    table = pd.DataFrame(
        [
            [df_explained, ss_explained, ms_explained, f_stat, p_value],
            [df_residual, ss_residual, ms_residual, None, None],
        ],
        columns=["df", "sum_sq", "mean_sq", "F", "PR(>F)"],
        index=("categories", "Residual"),
    )

    g_cat = np.array(group_stats_index)
    n_groups = len(g_cat)
    gnobs = group_stats_count
    gmeans = group_stats_sum / gnobs
    gvar = table.loc["Residual"]["mean_sq"] / gnobs
    g1, g2 = np.array(list(itertools.combinations(np.arange(n_groups), 2))).T

    mn = gmeans[g1] - gmeans[g2]
    se = np.sqrt(gvar[g1] + gvar[g2])
    tval = mn / se
    df = table.at["Residual", "df"]

    # psturng: studentized range distribution for Tukey
    pval = psturng(np.sqrt(2.0) * np.abs(tval), n_groups, df)

    thsd = pd.DataFrame(
        columns=[
            "A",
            "B",
            "mean(A)",
            "mean(B)",
            "diff",
            "Std.Err.",
            "t value",
            "Pr(>|t|)",
        ],
        index=range(n_groups * (n_groups - 1) // 2),
    )
    thsd["A"] = np.array(g_cat)[g1.astype(int)]
    thsd["B"] = np.array(g_cat)[g2.astype(int)]
    thsd["mean(A)"] = gmeans[g1]
    thsd["mean(B)"] = gmeans[g2]
    thsd["diff"] = mn
    thsd["Std.Err."] = se
    thsd["t value"] = tval
    thsd["Pr(>|t|)"] = pval

    # group means & sample stds
    variances = group_ssq / group_stats_count - gmeans**2
    sample_vars = (group_stats_count - 1.0) / group_stats_count * variances
    sample_stds = np.sqrt(sample_vars)

    transfer_ = {
        "n_obs": int(n_obs),
        "categories": categories,
        "means": gmeans.tolist(),
        "sample_stds": sample_stds.tolist(),
        "var_min_per_group": var_min_per_group.tolist(),
        "var_max_per_group": var_max_per_group.tolist(),
        "group_stats_index": group_stats_index,
        "df_explained": float(df_explained),
        "df_residual": float(df_residual),
        "ss_explained": float(ss_explained),
        "ss_residual": float(ss_residual),
        "ms_explained": float(ms_explained),
        "ms_residual": float(ms_residual),
        "f_stat": float(f_stat),
        "p_value": float(p_value),
        "thsd A": thsd["A"].tolist(),
        "thsd B": thsd["B"].tolist(),
        "thsd mean(A)": thsd["mean(A)"].tolist(),
        "thsd mean(B)": thsd["mean(B)"].tolist(),
        "thsd diff": thsd["diff"].tolist(),
        "thsd Std.Err.": thsd["Std.Err."].tolist(),
        "thsd t value": thsd["t value"].tolist(),
        "thsd Pr(>|t|)": thsd["Pr(>|t|)"].tolist(),
    }

    # Merge the summary table info
    transfer_.update(anova_table)
    return transfer_

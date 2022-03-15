import json
from typing import TypeVar

import pandas as pd
from pydantic import BaseModel

from mipengine.udfgen.udfgenerator import literal
from mipengine.udfgen.udfgenerator import merge_transfer
from mipengine.udfgen.udfgenerator import relation
from mipengine.udfgen.udfgenerator import transfer
from mipengine.udfgen.udfgenerator import udf


class AnovaResult(BaseModel):
    anova_table: dict
    tukey_test: list


def run(algo_interface):
    local_run = algo_interface.run_udf_on_local_nodes
    global_run = algo_interface.run_udf_on_global_node

    X_relation = algo_interface.initial_view_tables["x"]
    x_var_name = [
        x.__dict__["name"]
        for x in X_relation.get_table_schema().__dict__["columns"]
        if x.__dict__["name"] != "row_id"
    ].pop()

    covar_enums = list(algo_interface.metadata[x_var_name]["enumerations"])

    Y_relation = algo_interface.initial_view_tables["y"]
    local_transfers = local_run(
        func=local1,
        keyword_args=dict(y=Y_relation, x=X_relation, covar_enums=covar_enums),
        share_to_global=[True],
    )

    result = global_run(
        func=global1,
        keyword_args=dict(local_transfers=local_transfers),
    )

    result = json.loads(result.get_table_data()[1][0])
    n_obs = result["n_obs"]
    anova_result = {
        "df_residual": result["df_residual"],
        "df_explained": result["df_explained"],
        "ss_residual": result["ss_residual"],
        "ss_explained": result["ss_explained"],
        "ms_residual": result["ms_residual"],
        "ms_explained": result["ms_explained"],
        "p_value": result["p_value"],
        "f_stat": result["f_stat"],
    }

    tukey_result = {
        "groupA": result["thsd A"],
        "groupB": result["thsd B"],
        "meanA": result["thsd mean(A)"],
        "meanB": result["thsd mean(B)"],
        "diff": result["thsd diff"],
        "se": result["thsd Std.Err."],
        "t_stat": result["thsd t value"],
        "p_tukey": result["thsd Pr(>|t|)"],
    }
    df_tukey_result = pd.DataFrame(tukey_result)
    tukey_test = df_tukey_result.to_dict(orient="records")

    anova_table = AnovaResult(anova_table=anova_result, tukey_test=tukey_test)

    return anova_table


S = TypeVar("S")


@udf(
    y=relation(schema=S),
    x=relation(schema=S),
    covar_enums=literal(),
    return_type=[transfer()],
)
def local1(y, x, covar_enums):
    import pandas as pd

    variable = y.reset_index(drop=True).to_numpy().squeeze()
    covariable = x.reset_index(drop=True).to_numpy().squeeze()
    var_label = y.columns.values.tolist()[0]
    covar_label = x.columns.values.tolist()[0]
    dataset = pd.DataFrame()
    dataset[var_label] = variable
    dataset[covar_label] = covariable
    var_sq = var_label + "_sq"
    dataset[var_sq] = variable ** 2
    n_obs = y.shape[0]

    # get overall stats
    overall_stats = dataset[var_label].agg(["count", "sum"])
    overall_ssq = dataset[var_sq].sum()
    overall_stats = overall_stats.append(pd.Series(data=overall_ssq, index=["sum_sq"]))

    # get group stats
    group_stats = (
        dataset[[var_label, covar_label]].groupby(covar_label).agg(["count", "sum"])
    )
    group_stats.columns = ["count", "sum"]
    group_ssq = dataset[[var_sq, covar_label]].groupby(covar_label).sum()
    group_ssq.columns = ["sum_sq"]
    group_stats_df = pd.DataFrame(group_stats)
    group_stats_df["group_ssq"] = group_ssq

    transfer_ = {
        "n_obs": n_obs,
        "var_label": var_label,
        "covar_label": covar_label,
        var_label: dataset[var_label].tolist(),
        covar_label: dataset[covar_label].tolist(),
        "covar_enums": covar_enums,
        "overall_stats_sum": overall_stats["sum"].tolist(),
        "overall_stats_count": overall_stats["count"].tolist(),
        "overall_ssq": overall_ssq.item(),
        "overall_stats_index": overall_stats.index.tolist(),
        "group_stats_sum": group_stats_df["sum"].tolist(),
        "group_stats_count": group_stats_df["count"].tolist(),
        "group_stats_ssq": group_stats_df["group_ssq"].tolist(),
        "group_stats_df_index": group_stats_df.index.tolist(),
    }

    return transfer_


@udf(local_transfers=merge_transfer(), return_type=[transfer()])
def global1(local_transfers):
    import itertools

    import numpy as np
    import scipy.stats as st
    from statsmodels.stats.libqsturng import psturng

    var_label = [t["var_label"] for t in local_transfers][0]
    covar_label = [t["covar_label"] for t in local_transfers][0]
    n_obs = sum(t["n_obs"] for t in local_transfers)
    overall_stats_sum = sum(np.array(t["overall_stats_sum"]) for t in local_transfers)
    overall_stats_count = sum(
        np.array(t["overall_stats_count"]) for t in local_transfers
    )
    overall_ssq = sum(np.array(t["overall_ssq"]) for t in local_transfers)
    group_stats_sum = sum(np.array(t["group_stats_sum"]) for t in local_transfers)
    group_stats_count = sum(np.array(t["group_stats_count"]) for t in local_transfers)
    group_ssq = sum(np.array(t["group_stats_ssq"]) for t in local_transfers)
    group_stats_index_all = [t["group_stats_df_index"] for t in local_transfers]

    group_stats_index = group_stats_index_all[0]
    if len(group_stats_index_all) > 1:
        for x, y in itertools.combinations(
            group_stats_index_all, len(group_stats_index_all)
        ):
            group_stats_index = x
            diff = list(set(x) - set(y))
            if diff != []:
                group_stats_index.append(diff)

    categories = local_transfers[0]["covar_enums"]
    if len(categories) < 2:
        raise ValueError("Cannot perform Anova when there is only one level")

    df_explained = len(group_stats_index) - 1
    df_residual = n_obs - len(group_stats_index)
    ss_residual = overall_ssq - sum(group_stats_sum ** 2 / group_stats_count)
    overall_mean = overall_stats_sum / overall_stats_count
    ss_total = overall_ssq - overall_stats_sum ** 2 / overall_stats_count

    ss_explained = sum(
        (overall_mean - group_stats_sum / group_stats_count) ** 2 * group_stats_count
    )
    ms_explained = ss_explained / df_explained
    ms_residual = ss_residual / df_residual
    f_stat = ms_explained / ms_residual
    p_value = 1 - st.f.cdf(f_stat, df_explained, df_residual)

    # get anova table
    anova_table = {
        "fields": ["", "df", "sum sq", "mean sq", "F value", "Pr(>F)"],
        "data": [
            [
                # len(categories),
                # categories,
                df_explained,
                ss_explained,
                ms_explained,
                f_stat,
                p_value,
            ],
            [
                # "Residual",
                df_residual,
                ss_residual,
                ms_residual,
                None,
                None,
            ],
        ],
        "title": "Anova Summary",
    }

    table = pd.DataFrame(
        anova_table["data"],
        columns=["df", "sum_sq", "mean_sq", "F", "PR(>F)"],
        index=("categories", "Residual"),
    )

    table.loc["categories"]: {
        "df": df_explained,
        "sum_sq": ss_explained,
        "mean_sq": ms_explained,
        "F": f_stat,
        "PR('>F')": p_value,
    }

    table.loc["Residual"]: {
        "df": df_residual,
        "sum_sq": ss_residual,
        "mean_sq": ms_residual,
        "F": None,
        "PR('>F')": None,
    }

    # tukey data
    # pairwise tukey (model, covar_enums)

    g_cat = np.array(group_stats_index)
    n_groups = len(g_cat)
    gnobs = group_stats_count
    gmeans = group_stats_sum / group_stats_count

    gvar = table.loc["Residual"]["mean_sq"] / gnobs
    g1, g2 = np.array(list(itertools.combinations(np.arange(n_groups), 2))).T
    mn = gmeans[g1] - gmeans[g2]
    se = np.sqrt(gvar[g1] + gvar[g2])
    tval = mn / se
    df = table.at["Residual", "df"]
    # psturng replaced with scipy stats' studentized_range
    pval = psturng(np.sqrt(2) * np.abs(tval), n_groups, df)
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

    tukey_data = thsd
    tukey_hsd_table = {
        "fields": list(tukey_data.columns),
        "data": list([list(row) for row in tukey_data.values]),
        "title": "Tuckey Honest Significant Differences",
    }

    tukey_dict = []
    for _, row in tukey_data.iterrows():
        tukey_row = dict()
        tukey_row["groupA"] = row["A"]
        tukey_row["groupB"] = row["B"]
        tukey_row["meanA"] = row["mean(A)"]
        tukey_row["meanB"] = row["mean(B)"]
        tukey_row["diff"] = row["diff"]
        tukey_row["se"] = row["Std.Err."]
        tukey_row["t_stat"] = row["t value"]
        tukey_row["p_tukey"] = row["Pr(>|t|)"]
        tukey_dict.append(tukey_row)

    # mean_plot = create_mean_plot(
    #     model.group_stats, var_label, covar_label, covar_enums
    # )

    title = "Means plot: {v} ~ {c}".format(v=var_label, c=covar_label)
    means = group_stats_sum / group_stats_count
    variances = group_ssq / group_stats_count - means ** 2
    sample_vars = (group_stats_count - 1) / group_stats_count * variances
    sample_stds = np.sqrt(sample_vars)

    categories = [c for c in categories if c in group_stats_index]
    df1dict = {"categories": categories, "means": means, "sample_stds": sample_stds}
    df1 = pd.DataFrame(df1dict)
    # means = [means[cat] for cat in df1[categories]]
    # sample_stds = [sample_stds[cat] for cat in categories]
    data = [(m - s, m, m + s) for m, s in zip(means, sample_stds)]
    bokeh_table = {
        "title": title,
        "data": data,
        "categories": categories,
        "xname": covar_label,
        "yname": "95% CI: " + var_label,
    }
    transfer_ = {
        "title": "ANOVA one-way",
        "n_obs": n_obs,
        "df_explained": df_explained,
        "df_residual": df_residual,
        "ss_explained": ss_explained,
        "ss_residual": ss_residual,
        "ms_explained": ms_explained,
        "ms_residual": ms_residual,
        "f_stat": f_stat,
        "p_value": p_value,
        "thsd A": thsd["A"].tolist(),
        "thsd B": thsd["B"].tolist(),
        "thsd mean(A)": thsd["mean(A)"].tolist(),
        "thsd mean(B)": thsd["mean(B)"].tolist(),
        "thsd diff": thsd["diff"].tolist(),
        "thsd Std.Err.": thsd["Std.Err."].tolist(),
        "thsd t value": thsd["t value"].tolist(),
        "thsd Pr(>|t|)": thsd["Pr(>|t|)"].tolist(),
    }

    return transfer_

from typing import TypeVar

import pandas as pd
from pydantic import BaseModel

from mipengine.algorithm_specification import AlgorithmSpecification
from mipengine.algorithm_specification import InputDataSpecification
from mipengine.algorithm_specification import InputDataSpecifications
from mipengine.algorithm_specification import InputDataStatType
from mipengine.algorithm_specification import InputDataType
from mipengine.algorithms.algorithm import Algorithm
from mipengine.algorithms.helpers import get_transfer_data
from mipengine.exceptions import BadUserInput
from mipengine.udfgen import literal
from mipengine.udfgen import merge_transfer
from mipengine.udfgen import relation
from mipengine.udfgen import secure_transfer
from mipengine.udfgen import transfer
from mipengine.udfgen import udf


class AnovaResult(BaseModel):
    anova_table: dict
    tuckey_test: list
    min_max_per_group: dict
    ci_info: dict


class AnovaOneWayAlgorithm(Algorithm, algname="anova_oneway"):
    @classmethod
    def get_specification(cls):
        return AlgorithmSpecification(
            name=cls.algname,
            desc="Test the difference in the means of the dependent variable between two or more groups, when there is a single independent covariate.",
            label="One-way ANOVA",
            enabled=True,
            inputdata=InputDataSpecifications(
                x=InputDataSpecification(
                    label="Covariate (independent)",
                    desc="A unique nominal variable.",
                    types=[InputDataType.INT, InputDataType.TEXT],
                    stattypes=[InputDataStatType.NOMINAL],
                    notblank=True,
                    multiple=False,
                ),
                y=InputDataSpecification(
                    label="Variable (dependent)",
                    desc="A unique continuous variable.",
                    types=[InputDataType.REAL, InputDataType.INT],
                    stattypes=[InputDataStatType.NUMERICAL],
                    notblank=True,
                    multiple=False,
                ),
            ),
        )

    def get_variable_groups(self):
        return [self.variables.x, self.variables.y]

    def run(self, engine, data, metadata):
        local_run = engine.run_udf_on_local_nodes
        global_run = engine.run_udf_on_global_node

        X_relation, Y_relation = data

        [x_var_name] = self.variables.x
        [y_var_name] = self.variables.y

        covar_enums = list(metadata[x_var_name]["enumerations"])

        sec_local_transfer, local_transfers = local_run(
            func=local1,
            keyword_args=dict(y=Y_relation, x=X_relation, covar_enums=covar_enums),
            share_to_global=[True, True],
        )
        try:
            result = global_run(
                func=global1,
                keyword_args=dict(
                    sec_local_transfer=sec_local_transfer,
                    local_transfers=local_transfers,
                ),
            )
        except Exception as ex:
            # TODO https://team-1617704806227.atlassian.net/browse/MIP-682
            if "Cannot perform Anova one-way. Covariable has only one level." in str(
                ex
            ):
                raise BadUserInput(
                    "Cannot perform Anova one-way. Covariable has only one level."
                )
            raise ex

        result = get_transfer_data(result)
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
            means,
            sample_stds,
            categories,
            var_min_per_group,
            var_max_per_group,
            group_stats_index,
        )
        anova_table = AnovaResult(
            anova_table=anova_result,
            tuckey_test=tukey_test,
            min_max_per_group=min_max_per_group,
            ci_info=ci_info,
        )

        return anova_table


S = TypeVar("S")
T = TypeVar("T")


@udf(
    y=relation(schema=S),
    x=relation(schema=T),
    covar_enums=literal(),
    return_type=[secure_transfer(sum_op=True, min_op=True, max_op=True), transfer()],
)
def local1(y, x, covar_enums):
    import sys

    import numpy as np
    import pandas as pd

    y.reset_index(drop=True)
    x.reset_index(drop=True)
    variable = y.to_numpy().squeeze()
    covariable = x.to_numpy().squeeze()
    var_label = y.columns.values.tolist()[0]
    covar_label = x.columns.values.tolist()[0]
    covar_enums = np.array(covar_enums)
    dataset = pd.DataFrame({var_label: variable, covar_label: covariable}, copy=False)

    n_obs = y.shape[0]
    min_per_group = dataset.groupby([covar_label]).min()
    max_per_group = dataset.groupby([covar_label]).max()
    var_sq = var_label + "_sq"
    dataset[var_sq] = variable**2

    # get overall stats
    overall_stats = dataset[var_label].agg(["count", "sum"])
    overall_ssq = dataset[var_sq].sum()
    overall_stats = pd.concat(
        [overall_stats, pd.Series(data=overall_ssq, index=["sum_sq"])]
    )

    # get group stats
    group_stats = (
        dataset[[var_label, covar_label]].groupby(covar_label).agg(["count", "sum"])
    )
    group_stats.columns = ["count", "sum"]
    group_ssq = dataset[[var_sq, covar_label]].groupby(covar_label).sum()
    group_ssq.columns = ["sum_sq"]
    min_per_group.columns = ["min_per_group"]
    max_per_group.columns = ["max_per_group"]
    group_stats_df = pd.DataFrame(group_stats)
    group_stats_df["group_ssq"] = group_ssq
    group_stats_df["min_per_group"] = min_per_group
    group_stats_df["max_per_group"] = max_per_group
    diff = list(set(covar_enums) - set(group_stats.index))
    if diff:
        diff_df = pd.DataFrame(
            0,
            index=diff,
            columns=["count", "sum", "group_ssq", "min_per_group", "max_per_group"],
        )
        diff_df["min_per_group"] = sys.float_info.max
        diff_df["max_per_group"] = sys.float_info.min
        group_stats_df = pd.concat([group_stats, diff_df])

    group_stats_df["groups"] = pd.Categorical(
        group_stats_df.index, categories=covar_enums.tolist(), ordered=True
    )
    group_stats_df.sort_values("groups", inplace=True)

    sec_transfer_ = {}
    sec_transfer_["n_obs"] = {"data": n_obs, "operation": "sum", "type": "int"}
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
    sec_transfer_["group_stats_sum"] = {
        "data": group_stats_df["sum"].tolist(),
        "operation": "sum",
        "type": "float",
    }
    sec_transfer_["group_stats_count"] = {
        "data": group_stats_df["count"].tolist(),
        "operation": "sum",
        "type": "float",
    }
    sec_transfer_["group_stats_ssq"] = {
        "data": group_stats_df["group_ssq"].tolist(),
        "operation": "sum",
        "type": "float",
    }
    sec_transfer_["min_per_group"] = {
        "data": group_stats_df["min_per_group"].tolist(),
        "operation": "min",
        "type": "float",
    }
    sec_transfer_["max_per_group"] = {
        "data": group_stats_df["max_per_group"].tolist(),
        "operation": "max",
        "type": "float",
    }
    transfer_ = {
        "var_label": var_label,
        "covar_label": covar_label,
        "covar_enums": covar_enums.tolist(),
        "group_stats_df_index": group_stats_df.index.tolist(),
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
    import scipy.stats as st
    from statsmodels.stats.libqsturng import psturng

    n_obs = sec_local_transfer["n_obs"]
    var_label = [t["var_label"] for t in local_transfers][0]
    covar_label = [t["covar_label"] for t in local_transfers][0]
    var_min_per_group = np.array(sec_local_transfer["min_per_group"])
    var_max_per_group = np.array(sec_local_transfer["max_per_group"])
    overall_stats_sum = np.array(sec_local_transfer["overall_stats_sum"])
    overall_stats_count = np.array(sec_local_transfer["overall_stats_count"])
    overall_ssq = np.array(sec_local_transfer["overall_ssq"])
    group_stats_sum = np.array(sec_local_transfer["group_stats_sum"])
    group_stats_count = np.array(sec_local_transfer["group_stats_count"])
    group_ssq = np.array(sec_local_transfer["group_stats_ssq"])
    group_stats_index_all = [t["group_stats_df_index"] for t in local_transfers]

    group_stats_index = group_stats_index_all[0]
    if len(group_stats_index_all) > 1:
        for x, y in itertools.combinations(group_stats_index_all, 2):
            group_stats_index = x
            diff = list(set(x) - set(y))
            if diff != []:
                pd.concat([group_stats_index, diff])

    # remove zero count groups
    if not np.all(group_stats_count):
        df = pd.DataFrame(
            {
                "groups": group_stats_index,
                "count": group_stats_count,
                "ssq": group_ssq,
                "sum": group_stats_sum,
                "min": var_min_per_group,
                "max": var_max_per_group,
            },
            index=group_stats_index,
        )
        df = df[df["count"] != 0]
        group_stats_index = df["groups"].tolist()
        group_stats_count = np.array(df["count"])
        group_stats_sum = np.array(df["sum"])
        group_ssq = np.array(df["ssq"])
        var_min_per_group = np.array(df["min"])
        var_max_per_group = np.array(df["max"])

    categories = local_transfers[0]["covar_enums"]
    if len(categories) < 2 or len(group_stats_index) < 2:
        raise ValueError("Cannot perform Anova one-way. Covariable has only one level.")

    df_explained = len(group_stats_index) - 1
    df_residual = n_obs - len(group_stats_index)
    ss_residual = overall_ssq - sum(group_stats_sum**2 / group_stats_count)
    overall_mean = overall_stats_sum / overall_stats_count

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

    # tukey data
    # pairwise tukey (model, covar_enums)

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
        tukey_row["p_tuckey"] = row["Pr(>|t|)"]
        tukey_dict.append(tukey_row)

    # means = group_stats_sum / group_stats_count
    variances = group_ssq / group_stats_count - gmeans**2
    sample_vars = (group_stats_count - 1) / group_stats_count * variances
    sample_stds = np.sqrt(np.array(sample_vars))

    transfer_ = {
        "n_obs": n_obs,
        "categories": categories,
        "means": gmeans.tolist(),
        "sample_stds": sample_stds.tolist(),
        "var_min_per_group": var_min_per_group.tolist(),
        "var_max_per_group": var_max_per_group.tolist(),
        "group_stats_index": group_stats_index,
        "xname": covar_label,
        "yname": "95% CI: " + var_label,
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


def get_min_max_ci_info(
    means,
    sample_stds,
    categories,
    var_min_per_group,
    var_max_per_group,
    group_stats_index,
):
    categories = [c for c in categories if c in group_stats_index]
    df1_means_stds_dict = {
        "categories": categories,
        "sample_stds": list(sample_stds),
        "means": list(means),
    }
    df_min_max = {
        "categories": categories,
        "min": var_min_per_group,
        "max": var_max_per_group,
    }
    df1_means_stds = pd.DataFrame(df1_means_stds_dict, index=categories).drop(
        "categories", 1
    )
    df1_means_stds["m-s"] = list(
        df1_means_stds["means"] - df1_means_stds["sample_stds"]
    )
    df1_means_stds["m+s"] = list(
        df1_means_stds["means"] + df1_means_stds["sample_stds"]
    )

    min_max_per_group = df_min_max
    ci_info = df1_means_stds.to_dict()

    return min_max_per_group, ci_info

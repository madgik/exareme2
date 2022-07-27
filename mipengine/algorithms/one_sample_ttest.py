import json
from typing import TypeVar

import numpy
from pydantic import BaseModel

from mipengine.udfgen import secure_transfer
from mipengine.udfgen.udfgenerator import literal
from mipengine.udfgen.udfgenerator import relation
from mipengine.udfgen.udfgenerator import transfer
from mipengine.udfgen.udfgenerator import udf


class TtestResult(BaseModel):
    t_stat: float
    df: int
    std: float
    p: float
    mean_diff: float
    se_diff: float
    ci_upper: float
    ci_lower: float
    cohens_d: float


def run(algo_interface):
    local_run = algo_interface.run_udf_on_local_nodes
    global_run = algo_interface.run_udf_on_global_node
    alpha = algo_interface.algorithm_parameters["alpha"]
    alternative = algo_interface.algorithm_parameters["alt_hypothesis"]
    if "mu" in algo_interface.algorithm_parameters.keys():
        mu = algo_interface.algorithm_parameters["mu"]
    else:
        mu = None

    X_relation, *_ = algo_interface.create_primary_data_views(
        variable_groups=[algo_interface.y_variables],
    )

    [y_var_name] = algo_interface.y_variables

    sec_local_transfer = local_run(
        func=local_one_sample,
        keyword_args=dict(x=X_relation, mu=mu),
        share_to_global=[True],
    )

    result = global_run(
        func=global_one_sample,
        keyword_args=dict(
            sec_local_transfer=sec_local_transfer, alpha=alpha, alternative=alternative
        ),
    )

    result = json.loads(result.get_table_data()[1][0])
    one_sample_ttest_res = TtestResult(
        t_stat=result["t_stat"],
        df=result["df"],
        std=result["std"],
        p=result["p_value"],
        mean_diff=result["mean_diff"],
        se_diff=result["se_diff"],
        ci_upper=result["ci_upper"],
        ci_lower=result["ci_lower"],
        cohens_d=result["cohens_d"],
    )

    return one_sample_ttest_res


S = TypeVar("S")


@udf(
    x=relation(schema=S),
    mu=literal(),
    return_type=[secure_transfer(sum_op=True)],
)
def local_one_sample(x, mu):
    x = x.reset_index(drop=True).to_numpy().squeeze()
    n_obs = len(x)
    sum_x = sum(x)
    sqrd_x = sum(x**2)
    diff_x = sum(x - mu)
    diff_sqrd_x = sum((x - mu) ** 2)

    sec_transfer_ = {
        "n_obs": {"data": n_obs, "operation": "sum"},
        "sum_x": {"data": sum_x.item(), "operation": "sum"},
        "sqrd_x": {"data": sqrd_x.tolist(), "operation": "sum"},
        "diff_x": {"data": diff_x.tolist(), "operation": "sum"},
        "diff_sqrd_x": {"data": diff_sqrd_x.tolist(), "operation": "sum"},
    }

    return sec_transfer_


@udf(
    sec_local_transfer=secure_transfer(sum_op=True),
    alpha=literal(),
    alternative=literal(),
    mu=literal(),
    return_type=[transfer()],
)
def global_one_sample(sec_local_transfer, alpha, alternative, mu):
    from scipy.stats import t

    n_obs = sec_local_transfer["n_obs"]
    sum_x = sec_local_transfer["sum_x"]
    sqrd_sum = sec_local_transfer["sqrd_sum"]
    diff_sum = sec_local_transfer["diff_x"]
    diff_sqrd_sum = sec_local_transfer["diff_sqrd"]

    smpl_mean = sum_x / n_obs

    # population mean
    # Population mean = Sample mean + T*(Standard error of the mean)
    # if mu == None:
    mu = 0

    # devel_x = sqrd_sum - 2 * sum_x * mu + (mu ** 2) * n_obs
    # standard deviation of the difference between means
    # sd = numpy.sqrt(devel_x / (n_obs - 1))
    sd = numpy.sqrt((diff_sqrd_sum - (diff_sum**2 / n_obs)) / (n_obs - 1))

    # standard error of the difference between means
    sed = sd / numpy.sqrt(n_obs)

    # t-statistic
    t_stat = (smpl_mean - mu) / sed
    df = n_obs - 1

    # p-value for alternative = 'greater'
    if alternative == "greater":
        p = 1.0 - t.cdf(t_stat, df)
    # p-value for alternative = 'less'
    elif alternative == "less":
        p = 1.0 - t.cdf(-t_stat, df)
    # p-value for alternative = 'two-sided'
    else:
        p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0

    # Confidence intervals
    ci = t.interval(alpha=1 - alpha, df=n_obs - 1, loc=smpl_mean, scale=sed)

    # Cohen’s d
    cohens_d = (smpl_mean - mu) / sd

    mean_diff = sum_x - mu * n_obs

    transfer_ = {
        "t_stat": t_stat,
        "std": sd,
        "p_value": p,
        "df": df,
        "mean_diff": diff_sum,
        "se_diff": sed,
        "ci_upper": ci[1],
        "ci_lower": ci[0],
        "cohens_d": cohens_d,
    }

    return transfer_

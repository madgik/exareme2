import numpy
from pydantic import BaseModel

from mipengine.algorithms.algorithm import Algorithm
from mipengine.algorithms.helpers import get_transfer_data
from mipengine.udfgen import literal
from mipengine.udfgen import relation
from mipengine.udfgen import secure_transfer
from mipengine.udfgen import transfer
from mipengine.udfgen import udf


class TtestResult(BaseModel):
    n_obs: int
    t_stat: float
    df: int
    std: float
    p: float
    mean_diff: float
    se_diff: float
    ci_upper: float
    ci_lower: float
    cohens_d: float


class OnesampleTTestAlgorithm(Algorithm, algname="ttest_onesample"):
    def get_variable_groups(self):
        return [self.executor.y_variables]

    def run(self):
        local_run = self.executor.run_udf_on_local_nodes
        global_run = self.executor.run_udf_on_global_node
        alpha = self.executor.algorithm_parameters["alpha"]
        alternative = self.executor.algorithm_parameters["alt_hypothesis"]
        mu = self.executor.algorithm_parameters["mu"]

        X_relation = self.executor.data_model_views[0]

        sec_local_transfer = local_run(
            func=local_one_sample,
            keyword_args=dict(x=X_relation, mu=mu),
            share_to_global=[True],
        )

        result = global_run(
            func=global_one_sample,
            keyword_args=dict(
                sec_local_transfer=sec_local_transfer,
                alpha=alpha,
                alternative=alternative,
                mu=mu,
            ),
        )

        result = get_transfer_data(result)
        one_sample_ttest_res = TtestResult(
            n_obs=result["n_obs"],
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


@udf(
    x=relation(),
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
        "n_obs": {"data": n_obs, "operation": "sum", "type": "int"},
        "sum_x": {"data": sum_x.item(), "operation": "sum", "type": "float"},
        "sqrd_x": {"data": sqrd_x.tolist(), "operation": "sum", "type": "float"},
        "diff_x": {"data": diff_x.tolist(), "operation": "sum", "type": "float"},
        "diff_sqrd_x": {
            "data": diff_sqrd_x.tolist(),
            "operation": "sum",
            "type": "float",
        },
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
    sqrd_x = sec_local_transfer["sqrd_x"]
    diff_sum = sec_local_transfer["diff_x"]
    diff_sqrd_x = sec_local_transfer["diff_sqrd_x"]

    smpl_mean = sum_x / n_obs
    # standard deviation of the difference between means
    sd = numpy.sqrt((diff_sqrd_x - (diff_sum**2 / n_obs)) / (n_obs - 1))

    # standard error of the difference between means
    sed = sd / numpy.sqrt(n_obs)

    # t-statistic
    t_stat = (smpl_mean - mu) / sed
    df = n_obs - 1

    # Confidence intervals
    ci_lower, ci_upper = t.interval(
        alpha=1 - alpha, df=n_obs - 1, loc=smpl_mean, scale=sed
    )

    # p-value for alternative = 'greater'
    if alternative == "greater":
        p = 1.0 - t.cdf(t_stat, df)
        ci_upper = "Infinity"
    # p-value for alternative = 'less'
    elif alternative == "less":
        p = 1.0 - t.cdf(-t_stat, df)
        ci_lower = "-Infinity"
    # p-value for alternative = 'two-sided'
    else:
        p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0

    # Cohen’s d
    cohens_d = -(smpl_mean - mu) / sd

    transfer_ = {
        "n_obs": n_obs,
        "t_stat": t_stat,
        "std": sd,
        "p_value": p,
        "df": df,
        "mean_diff": sum_x / n_obs,
        "se_diff": sed,
        "ci_upper": ci_upper,
        "ci_lower": ci_lower,
        "cohens_d": cohens_d,
    }

    return transfer_

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
    t_stat: float
    df: int
    p: float
    mean_diff: float
    se_diff: float
    ci_upper: float
    ci_lower: float
    cohens_d: float


class IndependentTTestAlgorithm(Algorithm, algname="ttest_independent"):
    def get_variable_groups(self):
        return [self.executor.x_variables, self.executor.y_variables]

    def run(self):
        local_run = self.executor.run_udf_on_local_nodes
        global_run = self.executor.run_udf_on_global_node
        conf_lvl = self.executor.algorithm_parameters["confidence_lvl"]
        alternative = self.executor.algorithm_parameters["alt_hypothesis"]

        X_relation, Y_relation = self.executor.data_model_views

        sec_local_transfer = local_run(
            func=local_independent,
            keyword_args=dict(y=Y_relation, x=X_relation),
            share_to_global=[True],
        )

        result = global_run(
            func=global_independent,
            keyword_args=dict(
                sec_local_transfer=sec_local_transfer,
                conf_lvl=conf_lvl,
                alternative=alternative,
            ),
        )

        result = get_transfer_data(result)
        res = TtestResult(
            t_stat=result["t_stat"],
            df=result["df"],
            p=result["p"],
            mean_diff=result["mean_diff"],
            se_diff=result["se_diff"],
            ci_upper=result["ci_upper"],
            ci_lower=result["ci_lower"],
            cohens_d=result["cohens_d"],
        )

        return res


@udf(
    y=relation(),
    x=relation(),
    return_type=[secure_transfer(sum_op=True)],
)
def local_independent(x, y):
    x.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)
    x1, x2 = x.values.squeeze(), y.values.squeeze()
    x1_sum = sum(x1)
    x2_sum = sum(x2)
    n_obs_x1 = len(x)
    n_obs_x2 = len(y)
    x1_sqrd_sum = numpy.einsum("i,i->", x1, x1)
    x2_sqrd_sum = numpy.einsum("i,i->", x2, x2)

    sec_transfer_ = {
        "n_obs_x1": {"data": n_obs_x1, "operation": "sum", "type": "int"},
        "n_obs_x2": {"data": n_obs_x2, "operation": "sum", "type": "int"},
        "sum_x1": {"data": x1_sum.item(), "operation": "sum", "type": "float"},
        "sum_x2": {"data": x2_sum.item(), "operation": "sum", "type": "float"},
        "x1_sqrd_sum": {
            "data": x1_sqrd_sum.tolist(),
            "operation": "sum",
            "type": "float",
        },
        "x2_sqrd_sum": {
            "data": x2_sqrd_sum.tolist(),
            "operation": "sum",
            "type": "float",
        },
    }

    return sec_transfer_


@udf(
    sec_local_transfer=secure_transfer(sum_op=True),
    conf_lvl=literal(),
    alternative=literal(),
    return_type=[transfer()],
)
def global_independent(sec_local_transfer, conf_lvl, alternative):
    from scipy.stats import t

    n_obs_x1 = sec_local_transfer["n_obs_x1"]
    n_obs_x2 = sec_local_transfer["n_obs_x2"]
    sum_x1 = sec_local_transfer["sum_x1"]
    sum_x2 = sec_local_transfer["sum_x2"]
    x1_sqrd_sum = sec_local_transfer["x1_sqrd_sum"]
    x2_sqrd_sum = sec_local_transfer["x2_sqrd_sum"]

    n_obs = n_obs_x1 + n_obs_x2
    mean_x1 = sum_x1 / n_obs_x1
    mean_x2 = sum_x2 / n_obs_x2
    devel_x1 = x1_sqrd_sum - 2 * mean_x1 * sum_x1 + (mean_x1**2) * n_obs_x1
    devel_x2 = x2_sqrd_sum - 2 * sum_x2 * mean_x2 + (mean_x2**2) * n_obs_x2
    sd_x2 = numpy.sqrt(devel_x2 / (n_obs_x2 - 1))
    sd_x1 = numpy.sqrt(devel_x1 / (n_obs_x1 - 1))

    # standard error of the difference between means
    sed_x1 = sd_x1 / numpy.sqrt(n_obs_x1)
    sed_x2 = sd_x2 / numpy.sqrt(n_obs_x2)

    # standard error on the difference between the samples
    sed = numpy.sqrt(sed_x1**2.0 + sed_x2**2.0)
    # t-statistic
    t_stat = (mean_x1 - mean_x2) / sed
    df = n_obs_x1 + n_obs_x2 - 2

    # Difference of means
    diff_mean = mean_x1 - mean_x2

    # Confidence intervals !WARNING: The ci values are not tested. The code should not be modified, unless there is
    # a test for the new method.
    ci_lower, ci_upper = t.interval(alpha=1 - conf_lvl, df=df, loc=diff_mean, scale=sed)

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
    cohens_d = (mean_x1 - mean_x2) / numpy.sqrt((sd_x1**2 + sd_x2**2) / 2)

    transfer_ = {
        "t_stat": t_stat,
        "df": df,
        "p": p,
        "mean_diff": diff_mean,
        "se_diff": sed,
        "ci_upper": ci_upper,
        "ci_lower": ci_lower,
        "cohens_d": cohens_d,
    }

    return transfer_

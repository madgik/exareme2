import numpy
from pydantic import BaseModel

from exareme2.algorithms.exareme2.algorithm import Algorithm
from exareme2.algorithms.exareme2.algorithm import AlgorithmDataLoader
from exareme2.algorithms.exareme2.helpers import get_transfer_data
from exareme2.algorithms.exareme2.udfgen import literal
from exareme2.algorithms.exareme2.udfgen import relation
from exareme2.algorithms.exareme2.udfgen import secure_transfer
from exareme2.algorithms.exareme2.udfgen import state
from exareme2.algorithms.exareme2.udfgen import transfer
from exareme2.algorithms.exareme2.udfgen import udf
from exareme2.worker_communication import BadUserInput

ALGORITHM_NAME = "ttest_independent"


class IndependentTTestDataLoader(AlgorithmDataLoader, algname=ALGORITHM_NAME):
    def get_variable_groups(self):
        return [self._variables.x, self._variables.y]


class TtestResult(BaseModel):
    t_stat: float
    df: int
    p: float
    mean_diff: float
    se_diff: float
    ci_upper: str
    ci_lower: str
    cohens_d: float


class IndependentTTestAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self, data, metadata):
        local_run = self.engine.run_udf_on_local_workers
        global_run = self.engine.run_udf_on_global_worker
        alpha = self.algorithm_parameters["alpha"]

        alternative = self.algorithm_parameters["alt_hypothesis"]
        groupA = self.algorithm_parameters["groupA"]
        groupB = self.algorithm_parameters["groupB"]

        X_relation, Y_relation = data

        self.split_state_local, split_result_local = local_run(
            func=split_data_into_groups_local,
            keyword_args=dict(y=Y_relation, x=X_relation, groupA=groupA, groupB=groupB),
            share_to_global=[False, True],
        )

        split_res = global_run(
            func=split_data_into_groups_global,
            keyword_args=dict(
                sec_local_transfer=split_result_local,
            ),
        )

        res = get_transfer_data(split_res)
        n_x1 = res["n_x1"]
        n_x2 = res["n_x2"]

        if n_x1 == 0:
            raise BadUserInput(
                f"Not enough data in {groupA}. Please select a group with more data."
            )
        if n_x2 == 0:
            raise BadUserInput(
                f"Not enough data in {groupB}. Please select a group with more data."
            )

        sec_local_transfer = local_run(
            func=local_independent,
            keyword_args=dict(split_state=self.split_state_local),
            share_to_global=[True],
        )

        result = global_run(
            func=global_independent,
            keyword_args=dict(
                sec_local_transfer=sec_local_transfer,
                alpha=alpha,
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
    groupA=literal(),
    groupB=literal(),
    return_type=[state(), secure_transfer(sum_op=True)],
)
def split_data_into_groups_local(x, y, groupA, groupB):
    import numpy as np

    x = np.array(x)
    y = np.array(y)

    state = {}
    x1 = y[x == groupA]
    x2 = y[x == groupB]
    state["x1"] = x1
    state["x2"] = x2
    n_x1 = len(x1)
    n_x2 = len(x2)

    transfer = {
        "n_x1": {"data": n_x1, "operation": "sum", "type": "int"},
        "n_x2": {"data": n_x2, "operation": "sum", "type": "int"},
    }

    return state, transfer


@udf(
    sec_local_transfer=secure_transfer(sum_op=True),
    return_type=[transfer()],
)
def split_data_into_groups_global(sec_local_transfer):
    n_x1 = sec_local_transfer["n_x1"]
    n_x2 = sec_local_transfer["n_x2"]

    transfer = {"n_x1": n_x1, "n_x2": n_x2}

    return transfer


@udf(
    split_state=state(),
    return_type=[secure_transfer(sum_op=True)],
)
def local_independent(split_state):
    import numpy as np

    x1 = np.array(split_state["x1"])
    x2 = np.array(split_state["x2"])
    x1_sum = sum(x1)
    x2_sum = sum(x2)
    n_obs_x1 = len(x1)
    n_obs_x2 = len(x2)
    x1_sqrd_sum = numpy.einsum("i,i->", x1, x1)
    x2_sqrd_sum = numpy.einsum("i,i->", x2, x2)

    sec_transfer_ = {
        "n_obs_x1": {"data": n_obs_x1, "operation": "sum", "type": "int"},
        "n_obs_x2": {"data": n_obs_x2, "operation": "sum", "type": "int"},
        "sum_x1": {"data": x1_sum, "operation": "sum", "type": "float"},
        "sum_x2": {"data": x2_sum, "operation": "sum", "type": "float"},
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
    alpha=literal(),
    alternative=literal(),
    return_type=[transfer()],
)
def global_independent(sec_local_transfer, alpha, alternative):
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
    ci_lower, ci_upper = t.interval(alpha=1 - alpha, df=df, loc=diff_mean, scale=sed)

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
    cohens_d = (mean_x1 - mean_x2) / numpy.sqrt(
        ((n_obs_x1 - 1) * sd_x1**2 + (n_obs_x2 - 1) * sd_x2**2) / (n_obs - 2)
    )

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

from pydantic import BaseModel

from exaflow.algorithms.exaflow.algorithm import Algorithm
from exaflow.algorithms.exaflow.exaflow_registry import exaflow_udf
from exaflow.algorithms.exaflow.library.stats.stats import ttest_one_sample

ALGORITHM_NAME = "ttest_onesample"
DEFAULT_ALPHA = 0.05
DEFAULT_ALT = "two-sided"
DEFAULT_MU = 0.0


class TTestResult(BaseModel):
    n_obs: int
    t_stat: float
    df: int
    std: float
    p: float
    mean_diff: float
    se_diff: float
    ci_upper: str
    ci_lower: str
    cohens_d: float


class OneSampleTTestAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self, metadata):
        alpha = self.parameters.get("alpha", DEFAULT_ALPHA)
        alternative = self.parameters.get("alt_hypothesis", DEFAULT_ALT)
        mu = self.parameters.get("mu", DEFAULT_MU)

        results = self.engine.run_algorithm_udf(
            func=local_step,
            positional_args={
                "inputdata": self.inputdata.json(),
                "alpha": alpha,
                "alternative": alternative,
                "mu": mu,
            },
        )
        result = results[0]

        return TTestResult(
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


@exaflow_udf(with_aggregation_server=True)
def local_step(inputdata, agg_client, alpha, alternative, mu):
    from exaflow.algorithms.exaflow.data_loading import load_algorithm_dataframe

    if not inputdata.y or len(inputdata.y) != 1:
        raise ValueError("One-sample t-test requires exactly one variable in 'y'.")

    data = load_algorithm_dataframe(inputdata, dropna=True)
    sample = data[inputdata.y[0]].to_numpy(dtype=float, copy=False)

    return ttest_one_sample(
        agg_client=agg_client,
        sample=sample,
        mu=mu,
        alpha=alpha,
        alternative=alternative,
    )

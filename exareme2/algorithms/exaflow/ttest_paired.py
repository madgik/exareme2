from pydantic import BaseModel

from exareme2.algorithms.exaflow.algorithm import Algorithm
from exareme2.algorithms.exaflow.exaflow_registry import exaflow_udf
from exareme2.algorithms.exaflow.library.stats.stats import ttest_paired

ALGORITHM_NAME = "ttest_paired_exaflow_aggregator"
DEFAULT_ALPHA = 0.05
DEFAULT_ALT = "two-sided"


class TTestResult(BaseModel):
    t_stat: float
    df: int
    p: float
    mean_diff: float
    se_diff: float
    ci_upper: str
    ci_lower: str
    cohens_d: float


class PairedTTestAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self, metadata):
        alpha = self.parameters.get("alpha", DEFAULT_ALPHA)
        alternative = self.parameters.get("alt_hypothesis", DEFAULT_ALT)

        results = self.engine.run_algorithm_udf(
            func=local_step,
            positional_args={
                "inputdata": self.inputdata.json(),
                "alpha": alpha,
                "alternative": alternative,
            },
        )
        result = results[0]

        return TTestResult(
            t_stat=result["t_stat"],
            df=result["df"],
            p=result["p_value"],
            mean_diff=result["mean_diff"],
            se_diff=result["se_diff"],
            ci_upper=result["ci_upper"],
            ci_lower=result["ci_lower"],
            cohens_d=result["cohens_d"],
        )


@exaflow_udf(with_aggregation_server=True)
def local_step(inputdata, csv_paths, agg_client, alpha, alternative):
    from exareme2.algorithms.utils.inputdata_utils import fetch_data

    if not inputdata.x or not inputdata.y:
        raise ValueError("Paired t-test requires both 'x' and 'y' variables.")

    x_var = inputdata.x[0]
    y_var = inputdata.y[0]

    data = fetch_data(inputdata, csv_paths)

    sample_x = data[x_var].to_numpy(dtype=float, copy=False)
    sample_y = data[y_var].to_numpy(dtype=float, copy=False)

    return ttest_paired(
        agg_client=agg_client,
        sample_x=sample_x,
        sample_y=sample_y,
        alpha=alpha,
        alternative=alternative,
    )

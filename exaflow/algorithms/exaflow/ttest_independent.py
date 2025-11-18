from pydantic import BaseModel

from exaflow.algorithms.exaflow.algorithm import Algorithm
from exaflow.algorithms.exaflow.exaflow_registry import exaflow_udf
from exaflow.algorithms.exaflow.library.stats.stats import ttest_independent

ALGORITHM_NAME = "ttest_independent"
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


class IndependentTTestAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self, metadata):
        alpha = self.parameters.get("alpha", DEFAULT_ALPHA)
        alternative = self.parameters.get("alt_hypothesis", DEFAULT_ALT)
        group_a = self.parameters.get("groupA")
        group_b = self.parameters.get("groupB")
        use_duckdb = True

        if group_a is None or group_b is None:
            raise ValueError(
                "Independent t-test requires 'groupA' and 'groupB' parameters."
            )

        results = self.engine.run_algorithm_udf(
            func=local_step,
            positional_args={
                "inputdata": self.inputdata.json(),
                "alpha": alpha,
                "alternative": alternative,
                "group_a": group_a,
                "group_b": group_b,
                "use_duckdb": use_duckdb,
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
def local_step(
    inputdata, csv_paths, agg_client, alpha, alternative, group_a, group_b, use_duckdb
):
    from exaflow.algorithms.exaflow.data_loading import load_algorithm_dataframe

    if not inputdata.x or not inputdata.y:
        raise ValueError(
            "Independent t-test requires both grouping ('x') and measurement ('y') variables."
        )

    group_var = inputdata.x[0]
    value_var = inputdata.y[0]

    data = load_algorithm_dataframe(inputdata, csv_paths, dropna=True)

    grouping = data[group_var]
    if hasattr(grouping, "ndim") and grouping.ndim > 1:
        # Some backends return a single-column DataFrame; convert to Series.
        grouping = grouping.squeeze()
    values = data[value_var]

    mask_a = grouping == group_a
    mask_b = grouping == group_b
    if hasattr(mask_a, "ndim") and mask_a.ndim > 1:
        mask_a = mask_a.squeeze()
    if hasattr(mask_b, "ndim") and mask_b.ndim > 1:
        mask_b = mask_b.squeeze()

    sample_a = values[mask_a].to_numpy(dtype=float, copy=False)
    sample_b = values[mask_b].to_numpy(dtype=float, copy=False)

    return ttest_independent(
        agg_client=agg_client,
        sample_a=sample_a,
        sample_b=sample_b,
        alpha=alpha,
        alternative=alternative,
    )

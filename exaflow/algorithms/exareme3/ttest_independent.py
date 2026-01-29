from exaflow.algorithms.exareme3.algorithm import Algorithm
from exaflow.algorithms.exareme3.exareme3_registry import exareme3_udf
from exaflow.algorithms.exareme3.library.stats.stats import ttest_independent
from exaflow.algorithms.exareme3.library.ttest_common import build_basic_ttest_result

ALGORITHM_NAME = "ttest_independent"


class IndependentTTestAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self, metadata):
        alpha = self.parameters.get("alpha")
        alternative = self.parameters.get("alt_hypothesis")
        group_a = self.parameters.get("groupA")
        group_b = self.parameters.get("groupB")

        results = self.run_local_udf(
            func=local_step,
            kw_args={
                "group_var": self.inputdata.x[0],
                "value_var": self.inputdata.y[0],
                "alpha": alpha,
                "alternative": alternative,
                "group_a": group_a,
                "group_b": group_b,
            },
        )
        return build_basic_ttest_result(results[0])


@exareme3_udf(with_aggregation_server=True)
def local_step(
    agg_client, data, group_var, value_var, alpha, alternative, group_a, group_b
):
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

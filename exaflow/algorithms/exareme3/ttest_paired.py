from exaflow.algorithms.exareme3.algorithm import Algorithm
from exaflow.algorithms.exareme3.exareme3_registry import exareme3_udf
from exaflow.algorithms.exareme3.library.stats.stats import ttest_paired
from exaflow.algorithms.exareme3.library.ttest_common import build_basic_ttest_result

ALGORITHM_NAME = "ttest_paired"


class PairedTTestAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self, metadata):
        alpha = self.parameters.get("alpha")
        alternative = self.parameters.get("alt_hypothesis")

        results = self.engine.run_algorithm_udf(
            func=local_step,
            positional_args={
                "x_var": self.inputdata.x[0],
                "y_var": self.inputdata.y[0],
                "alpha": alpha,
                "alternative": alternative,
            },
        )
        return build_basic_ttest_result(results[0])


@exareme3_udf(with_aggregation_server=True)
def local_step(agg_client, data, x_var, y_var, alpha, alternative):
    sample_x = data[x_var].to_numpy(dtype=float, copy=False)
    sample_y = data[y_var].to_numpy(dtype=float, copy=False)

    return ttest_paired(
        agg_client=agg_client,
        sample_x=sample_x,
        sample_y=sample_y,
        alpha=alpha,
        alternative=alternative,
    )

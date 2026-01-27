from exaflow.algorithms.exareme3.algorithm import Algorithm
from exaflow.algorithms.exareme3.exareme3_registry import exareme3_udf
from exaflow.algorithms.exareme3.library.stats.stats import ttest_one_sample
from exaflow.algorithms.exareme3.library.ttest_common import (
    build_one_sample_ttest_result,
)

ALGORITHM_NAME = "ttest_onesample"


class OneSampleTTestAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self, metadata):
        alpha = self.parameters.get("alpha")
        alternative = self.parameters.get("alt_hypothesis")
        mu = self.parameters.get("mu")

        results = self.engine.run_algorithm_udf(
            func=local_step,
            positional_args={
                "inputdata": self.inputdata.json(),
                "alpha": alpha,
                "alternative": alternative,
                "mu": mu,
            },
        )
        return build_one_sample_ttest_result(results[0])


@exareme3_udf(with_aggregation_server=True)
def local_step(data, inputdata, agg_client, alpha, alternative, mu):
    sample = data[inputdata.y[0]].to_numpy(dtype=float, copy=False)

    return ttest_one_sample(
        agg_client=agg_client,
        sample=sample,
        mu=mu,
        alpha=alpha,
        alternative=alternative,
    )

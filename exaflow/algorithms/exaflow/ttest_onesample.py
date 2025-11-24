from exaflow.algorithms.exaflow.algorithm import Algorithm
from exaflow.algorithms.exaflow.exaflow_registry import exaflow_udf
from exaflow.algorithms.exaflow.library.stats.stats import ttest_one_sample
from exaflow.algorithms.exaflow.library.ttest_common import DEFAULT_ALPHA
from exaflow.algorithms.exaflow.library.ttest_common import DEFAULT_ALT
from exaflow.algorithms.exaflow.library.ttest_common import (
    OneSampleTTestResult as TTestResult,
)
from exaflow.algorithms.exaflow.library.ttest_common import (
    build_one_sample_ttest_result,
)

ALGORITHM_NAME = "ttest_onesample"
DEFAULT_MU = 0.0


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
        return build_one_sample_ttest_result(results[0])


@exaflow_udf(with_aggregation_server=True)
def local_step(data, inputdata, agg_client, alpha, alternative, mu):

    if not inputdata.y or len(inputdata.y) != 1:
        raise ValueError("One-sample t-test requires exactly one variable in 'y'.")

    sample = data[inputdata.y[0]].to_numpy(dtype=float, copy=False)

    return ttest_one_sample(
        agg_client=agg_client,
        sample=sample,
        mu=mu,
        alpha=alpha,
        alternative=alternative,
    )

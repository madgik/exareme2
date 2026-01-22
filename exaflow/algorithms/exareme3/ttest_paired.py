from exaflow.algorithms.exareme3.algorithm import Algorithm
from exaflow.algorithms.exareme3.exareme3_registry import exaflow_udf
from exaflow.algorithms.exareme3.library.stats.stats import ttest_paired
from exaflow.algorithms.exareme3.library.ttest_common import DEFAULT_ALPHA
from exaflow.algorithms.exareme3.library.ttest_common import DEFAULT_ALT
from exaflow.algorithms.exareme3.library.ttest_common import TTestResult
from exaflow.algorithms.exareme3.library.ttest_common import build_basic_ttest_result

ALGORITHM_NAME = "ttest_paired"


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
        return build_basic_ttest_result(results[0])


@exaflow_udf(with_aggregation_server=True)
# TODO codex here i would like to have a logic of adding an anotations @duck_data_loading or @csvs_data_loading  if duckdb the dataframe that will be using the usual load_algorithm_dataframe the same for csvs you can see the csvs loading at /home/kfilippopolitis/Desktop/exaflow/exaflow/algorithms/utils on file inputdata_utils this should be moved in the worker.
def local_step(data, inputdata, agg_client, alpha, alternative):

    if not inputdata.x or not inputdata.y:
        raise ValueError("Paired t-test requires both 'x' and 'y' variables.")

    x_var = inputdata.x[0]
    y_var = inputdata.y[0]

    sample_x = data[x_var].to_numpy(dtype=float, copy=False)
    sample_y = data[y_var].to_numpy(dtype=float, copy=False)

    return ttest_paired(
        agg_client=agg_client,
        sample_x=sample_x,
        sample_y=sample_y,
        alpha=alpha,
        alternative=alternative,
    )

from exareme2.algorithms.exaflow.algorithm import Algorithm
from exareme2.algorithms.exaflow.exaflow_registry import exaflow_udf
from exareme2.worker_communication import ColumnDataFloat
from exareme2.worker_communication import ColumnDataStr
from exareme2.worker_communication import TabularDataResult

ALGORITHM_NAME = "standard_deviation"


class StandardDeviationAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self, metadata):
        # This call runs the local UDF on all workers.
        # They all execute the aggregation server calls so each one returns the same final standard deviation.
        std_deviation = self.engine.run_algorithm_udf_with_aggregator(
            func=local_step,
            positional_args={"inputdata": self.inputdata.json()},
        )

        # Build the result: for each column in y (here, assume a single column scenario), assign the computed std.
        result = TabularDataResult(
            title="Standard Deviation",
            columns=[
                ColumnDataStr(name="variable", data=self.inputdata.y),
                ColumnDataFloat(name="std_deviation", data=[std_deviation]),
            ],
        )
        return result


def compute_stddev(agg_client, data):
    import numpy as np

    n = len(data)
    if n <= 1:
        return 0.0
    total = agg_client.sum([float(np.sum(data))])[0]
    mean = total / n
    # One remote call instead of two:
    sum_sq = agg_client.sum([float(np.sum(np.square(data)))])[0]
    variance = (sum_sq / n) - mean**2
    return float(np.sqrt(max(variance, 0.0)))


@exaflow_udf
def local_step(inputdata, csv_paths, agg_client):
    from exareme2.algorithms.utils.inputdata_utils import fetch_data

    data = fetch_data(inputdata, csv_paths)
    return compute_stddev(agg_client, data[inputdata.y[0]])

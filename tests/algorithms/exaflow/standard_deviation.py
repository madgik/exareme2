from typing import List

from exareme2.aggregator.aggregator_client import AggregationClient
from exareme2.algorithms.exaflow.algorithm import Algorithm
from exareme2.algorithms.exaflow.exaflow_registry import exaflow_udf
from exareme2.worker_communication import ColumnDataFloat
from exareme2.worker_communication import ColumnDataStr
from exareme2.worker_communication import TabularDataResult

ALGORITHM_NAME = "standard_deviation"


class StandardDeviationAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self, metadata):
        # This call runs the local UDF on all workers.
        # They all execute the aggregator calls so each one returns the same final standard deviation.
        std_deviation = self.engine.run_algorithm_udf_with_aggregator(
            func="standard_deviation_local_step",
            positional_args={"inputdata": self.inputdata.json()},
            on_monetdb=True,
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


@exaflow_udf
def local_step(data: List[float]) -> float:
    import math

    from exareme2.aggregator.aggregator_client import AggregationClient

    # Create an AggregationClient instance with a fixed request ID.
    agg_client = AggregationClient("<here goes the request_id>")

    # 'data' is received as a Python list of floats.
    # Filter out any NaN values.
    clean_data = [xi for xi in data if not math.isnan(xi)]

    if agg_client.count(clean_data) <= 1:
        return 0

    # Compute E(x^2): the average of squared valid values.
    avg_x_squared = agg_client.avg([float(xi) ** 2 for xi in clean_data])
    # Compute E(x): the average of the valid values.
    avg_x = agg_client.avg(clean_data)

    # Standard deviation: sqrt(E(x^2) - (E(x))^2)
    return math.sqrt(avg_x_squared - avg_x**2)


# def compute_stddev(agg_client, data):
#     import math
#
#     count = agg_client.count(data)
#     if count <= 1:
#         return 0
#     avg_x_squared = agg_client.avg([float(xi) ** 2 for xi in data])
#     avg_x = agg_client.avg(data)
#     return math.sqrt(avg_x_squared - avg_x**2)
#
#
# @exaflow_udf
# def local_step(request_id, inputdata, csv_paths):
#     from exareme2.algorithms.utils.inputdata_utils import fetch_data
#
#     data = fetch_data(inputdata, csv_paths)
#     agg_client = AggregationClient(request_id=request_id)
#     return compute_stddev(agg_client, data[inputdata.y[0]])

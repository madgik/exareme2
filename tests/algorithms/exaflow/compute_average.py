from exaflow.algorithms.exaflow.algorithm import Algorithm
from exaflow.algorithms.exaflow.exaflow_registry import exaflow_udf
from exaflow.worker_communication import ColumnDataFloat
from exaflow.worker_communication import ColumnDataStr
from exaflow.worker_communication import TabularDataResult

ALGORITHM_NAME = "compute_average"


class ComputeAverage(Algorithm, algname=ALGORITHM_NAME):
    def run(self, metadata):
        local_results = self.engine.run_algorithm_udf(
            func=local_step,
            positional_args={"inputdata": self.inputdata.json()},
        )

        results = {}
        for column in self.inputdata.y:
            global_sum = sum(res[column]["sum"] for res in local_results)
            global_count = sum(res[column]["count"] for res in local_results)
            average = global_sum / global_count if global_count else float("nan")
            results[column] = average

        # Prepare result columns
        variables = list(results.keys())
        averages = list(results.values())

        return TabularDataResult(
            title="Average",
            columns=[
                ColumnDataStr(name="variable", data=variables),
                ColumnDataFloat(name="average", data=averages),
            ],
        )


def compute_local_sum_and_count(columns, data):
    results = {}
    for column in columns:
        col_data = data[column]
        results[column] = {
            "sum": float(col_data.sum()),
            "count": int(col_data.count()),
        }
    return results


@exaflow_udf
def local_step(inputdata, csv_paths):
    from exaflow.algorithms.utils.inputdata_utils import fetch_data

    data = fetch_data(inputdata, csv_paths)
    columns = inputdata.y
    return compute_local_sum_and_count(columns, data)

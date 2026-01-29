from exaflow.algorithms.exareme3.algorithm import Algorithm
from exaflow.algorithms.exareme3.exareme3_registry import exareme3_udf
from exaflow.worker_communication import ColumnDataFloat
from exaflow.worker_communication import ColumnDataStr
from exaflow.worker_communication import TabularDataResult

ALGORITHM_NAME = "compute_average"


class ComputeAverage(Algorithm, algname=ALGORITHM_NAME):
    def run(self, metadata):
        local_results = self.run_local_udf(
            func=local_step,
            kw_args={},
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


@exareme3_udf
def local_step(data, y_vars):
    return compute_local_sum_and_count(y_vars, data)

from typing import List
from typing import Sequence

from pydantic import BaseModel

from exareme2.algorithms.exaflow.algorithm import Algorithm
from exareme2.algorithms.exaflow.exaflow_registry import exaflow_udf
from exareme2.algorithms.exaflow.library.stats.stats import pearson_correlation

ALGORITHM_NAME = "pearson_correlation_exaflow_aggregator"
DEFAULT_ALPHA = 0.95


class PearsonResult(BaseModel):
    title: str
    n_obs: int
    correlations: dict
    p_values: dict
    ci_hi: dict
    ci_lo: dict


class PearsonCorrelationAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self, metadata):
        alpha = self.parameters.get("alpha", DEFAULT_ALPHA)
        results = self.engine.run_algorithm_udf(
            func=local_step,
            positional_args={
                "inputdata": self.inputdata.json(),
                "alpha": alpha,
            },
        )
        result = results[0]

        x_vars = self.inputdata.x or self.inputdata.y or []
        y_vars = self.inputdata.y or []
        if not x_vars or not y_vars:
            raise ValueError("Pearson correlation requires at least one variable.")

        corr_dict, p_dict, ci_hi_dict, ci_lo_dict = _format_result_matrices(
            result,
            row_names=x_vars,
            column_names=y_vars,
        )

        return PearsonResult(
            title="Pearson Correlation Coefficient",
            n_obs=result["n_obs"],
            correlations=corr_dict,
            p_values=p_dict,
            ci_hi=ci_hi_dict,
            ci_lo=ci_lo_dict,
        )


def _format_result_matrices(
    result, *, row_names: Sequence[str], column_names: Sequence[str]
):
    correlations = result["correlations"]
    p_values = result["p_values"]
    ci_hi = result["ci_hi"]
    ci_lo = result["ci_lo"]

    def _build_matrix_dict(values_matrix):
        matrix_dict = {"variables": list(row_names)}
        matrix_dict.update({col: row for col, row in zip(column_names, values_matrix)})
        return matrix_dict

    return (
        _build_matrix_dict(correlations),
        _build_matrix_dict(p_values),
        _build_matrix_dict(ci_hi),
        _build_matrix_dict(ci_lo),
    )


@exaflow_udf(with_aggregation_server=True)
def local_step(inputdata, csv_paths, agg_client, alpha):
    from exareme2.worker.exaflow.duckdb import pearson_correlation as duckdb_pearson

    return duckdb_pearson.run_pearson(inputdata, agg_client, alpha)

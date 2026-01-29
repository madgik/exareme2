from typing import Sequence

from pydantic import BaseModel

from exaflow.algorithms.exareme3.algorithm import Algorithm
from exaflow.algorithms.exareme3.exareme3_registry import exareme3_udf
from exaflow.algorithms.exareme3.library.stats.stats import pearson_correlation

ALGORITHM_NAME = "pearson_correlation"


class PearsonResult(BaseModel):
    title: str
    n_obs: int
    correlations: dict
    p_values: dict
    ci_hi: dict
    ci_lo: dict


class PearsonCorrelationAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self):
        alpha = self.get_parameter("alpha")
        if self.inputdata.x:
            x_vars = self.inputdata.x
        else:
            x_vars = self.inputdata.y
        results = self.run_local_udf(
            func=local_step,
            kw_args={
                "y_vars": self.inputdata.y,
                "x_vars": x_vars,
                "alpha": alpha,
            },
        )
        result = results[0]

        x_vars = self.inputdata.x or self.inputdata.y
        y_vars = self.inputdata.y

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


@exareme3_udf(with_aggregation_server=True)
def local_step(agg_client, data, y_vars, x_vars, alpha):
    # Use numpy arrays directly to avoid pandas alignment overhead.
    x_matrix = data[x_vars].to_numpy(dtype=float, copy=False)
    y_matrix = data[y_vars].to_numpy(dtype=float, copy=False)

    return pearson_correlation(
        agg_client=agg_client,
        x=x_matrix,
        y=y_matrix,
        alpha=alpha,
    )

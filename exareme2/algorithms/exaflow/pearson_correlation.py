import warnings

import numpy
from pydantic import BaseModel

from exareme2.aggregator.constants import AggregationType
from exareme2.algorithms.exaflow.aggregator_client import AggregationClient
from exareme2.algorithms.exaflow.algorithm import Algorithm
from exareme2.algorithms.exaflow.exaflow_registry import exaflow_udf

ALGORITHM_NAME = "pearson_correlation"


class PearsonResult(BaseModel):
    title: str
    n_obs: int
    correlations: dict
    p_values: dict
    ci_hi: dict
    ci_lo: dict


class PearsonCorrelationAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self, metadata):
        result = self.engine.run_algorithm_udf_with_aggregator(
            func="pearson_correlation_local_step",
            positional_args={"inputdata": self.inputdata.json()},
        )

        n_obs = result["n_obs"]

        corr_dict, p_values_dict, ci_hi_dict, ci_lo_dict = create_dicts(
            result, self.inputdata.x.columns, self.inputdata.y.columns
        )

        result = PearsonResult(
            title="Pearson Correlation Coefficient",
            n_obs=n_obs,
            correlations=corr_dict,
            p_values=p_values_dict,
            ci_hi=ci_hi_dict,
            ci_lo=ci_lo_dict,
        )
        return result


def compute_pearson_correlation(agg_client, x, y):
    n_obs = y.shape[0]
    Y = y.to_numpy()
    X = Y if x is None else x.to_numpy()

    warnings.warn(str(Y))
    warnings.warn(str(X))
    n_obs = agg_client.aggregate(AggregationType.SUM, n_obs)
    sx = agg_client.aggregate(AggregationType.SUM, numpy.einsum("ij->j", X))
    sy = agg_client.aggregate(AggregationType.SUM, numpy.einsum("ij->j", Y))
    sxx = agg_client.aggregate(AggregationType.SUM, numpy.einsum("ij,ij->j", X, X))
    sxy = agg_client.aggregate(AggregationType.SUM, numpy.einsum("ji,jk->ki", X, Y))
    syy = agg_client.aggregate(AggregationType.SUM, numpy.einsum("ij,ij->j", Y, Y))

    import scipy.special as special
    import scipy.stats as st

    sx = numpy.array(sx)
    sy = numpy.array(sy)
    sxx = numpy.array(sxx)
    sxy = numpy.array(sxy)
    syy = numpy.array(syy)

    df = n_obs - 2

    d = (
        numpy.sqrt(n_obs * sxx - sx * sx)
        * numpy.sqrt(n_obs * syy - sy * sy)[:, numpy.newaxis]
    )
    correlations = (n_obs * sxy - sx * sy[:, numpy.newaxis]) / d
    correlations[d == 0] = 0
    correlations = correlations.clip(-1, 1)
    t_squared = correlations**2 * (df / ((1.0 - correlations) * (1.0 + correlations)))
    p_values = special.betainc(
        0.5 * df, 0.5, numpy.fmin(numpy.asarray(df / (df + t_squared)), 1.0)
    )
    p_values[abs(correlations) == 1] = 0
    r_z = numpy.arctanh(correlations)
    se = 1 / numpy.sqrt(n_obs - 3)
    alpha = 0.6394327706281919  # needs to be taken from parameters
    z = st.norm.ppf(1 - alpha / 2)
    lo_z, hi_z = r_z - z * se, r_z + z * se
    ci_lo, ci_hi = numpy.tanh((lo_z, hi_z))
    return {
        "n_obs": n_obs,
        "correlations": correlations.tolist(),
        "p_values": p_values.tolist(),
        "ci_lo": ci_lo.tolist(),
        "ci_hi": ci_hi.tolist(),
    }


def create_dicts(global_result, row_names, column_names):
    correlations = global_result["correlations"]
    p_values = global_result["p_values"]
    ci_hi = global_result["ci_hi"]
    ci_lo = global_result["ci_lo"]

    corr_dict = {}
    corr_dict["variables"] = row_names
    corr_dict.update({key: value for key, value in zip(column_names, correlations)})

    p_values_dict = {}
    p_values_dict["variables"] = row_names
    p_values_dict.update({key: value for key, value in zip(column_names, p_values)})

    ci_hi_dict = {}
    ci_hi_dict["variables"] = row_names
    ci_hi_dict.update({key: value for key, value in zip(column_names, ci_hi)})

    ci_lo_dict = {}
    ci_lo_dict["variables"] = row_names
    ci_lo_dict.update({key: value for key, value in zip(column_names, ci_lo)})

    return corr_dict, p_values_dict, ci_hi_dict, ci_lo_dict


@exaflow_udf
def local_step(request_id, inputdata, csv_paths):
    from exareme2.algorithms.utils.inputdata_utils import fetch_data

    data = fetch_data(inputdata, csv_paths)
    agg_client = AggregationClient(request_id=request_id)
    return compute_pearson_correlation(agg_client, data[inputdata.x], data[inputdata.y])

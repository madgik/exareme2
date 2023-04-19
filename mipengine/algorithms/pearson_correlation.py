from typing import TypeVar

import numpy
from pydantic import BaseModel

from mipengine.algorithm_specification import AlgorithmSpecification
from mipengine.algorithm_specification import InputDataSpecification
from mipengine.algorithm_specification import InputDataSpecifications
from mipengine.algorithm_specification import InputDataStatType
from mipengine.algorithm_specification import InputDataType
from mipengine.algorithm_specification import ParameterSpecification
from mipengine.algorithm_specification import ParameterType
from mipengine.algorithms.algorithm import Algorithm
from mipengine.algorithms.algorithm import AlgorithmDataLoader
from mipengine.algorithms.helpers import get_transfer_data
from mipengine.udfgen import literal
from mipengine.udfgen import relation
from mipengine.udfgen import secure_transfer
from mipengine.udfgen import transfer
from mipengine.udfgen import udf

ALGORITHM_NAME = "pearson_correlation"


class PearsonCorrelationDataLoader(AlgorithmDataLoader, algname=ALGORITHM_NAME):
    def get_variable_groups(self):
        if self._variables.x:
            variable_groups = [self._variables.x, self._variables.y]
        else:
            variable_groups = [self._variables.y, self._variables.y]
        return variable_groups


class PearsonResult(BaseModel):
    title: str
    n_obs: int
    correlations: dict
    p_values: dict
    ci_hi: dict
    ci_lo: dict


class PearsonCorrelationAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    @classmethod
    def get_specification(cls):
        return AlgorithmSpecification(
            name=cls.algname,
            desc="Measure the linear relationship between two continuous variables. It calculates the correlation coefficient (range: -1 to 1). The correlation matrix will be computed between all possible pairs of variables and covariates. Leaving covariates empty is equivalent to having covariates = variables.",
            label="Pearson Correlation Matrix",
            enabled=True,
            inputdata=InputDataSpecifications(
                y=InputDataSpecification(
                    label="Variables",
                    desc="Nuerical variables on x axis of correlation matrix.",
                    types=[InputDataType.REAL, InputDataType.INT],
                    stattypes=[InputDataStatType.NUMERICAL],
                    notblank=True,
                    multiple=True,
                ),
                x=InputDataSpecification(
                    label="Covariates (optional)",
                    desc="Nuerical variables on y axis of correlation matrix.",
                    types=[InputDataType.REAL, InputDataType.INT],
                    stattypes=[InputDataStatType.NUMERICAL],
                    notblank=False,
                    multiple=True,
                ),
            ),
            parameters={
                "alpha": ParameterSpecification(
                    label="Confidence level",
                    desc="The confidence level α used in the calculation of the confidence intervals for the correlation coefficients.",
                    types=[ParameterType.REAL],
                    notblank=True,
                    multiple=False,
                    default=0.95,
                    min=0.0,
                    max=1.0,
                ),
            },
        )

    def run(self, data, metadata):
        local_run = self._engine.run_udf_on_local_nodes
        global_run = self._engine.run_udf_on_global_node
        alpha = self.algorithm_parameters["alpha"]

        X_relation, Y_relation = data

        local_transfers = local_run(
            func=local1,
            keyword_args=dict(y=Y_relation, x=X_relation),
            share_to_global=[True],
        )

        result = global_run(
            func=global1,
            keyword_args=dict(local_transfers=local_transfers, alpha=alpha),
        )

        result = get_transfer_data(result)
        n_obs = result["n_obs"]

        corr_dict, p_values_dict, ci_hi_dict, ci_lo_dict = create_dicts(
            result, X_relation.columns, Y_relation.columns
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


S = TypeVar("S")


@udf(
    y=relation(schema=S),
    x=relation(schema=S),
    return_type=[secure_transfer(sum_op=True)],
)
def local1(y, x):
    n_obs = y.shape[0]
    Y = y.to_numpy()
    X = Y if x is None else x.to_numpy()

    sx = numpy.einsum("ij->j", X)
    sy = numpy.einsum("ij->j", Y)
    sxx = numpy.einsum("ij,ij->j", X, X)
    sxy = numpy.einsum("ji,jk->ki", X, Y)
    syy = numpy.einsum("ij,ij->j", Y, Y)

    transfer_ = {}
    transfer_["n_obs"] = {"data": n_obs, "operation": "sum", "type": "int"}
    transfer_["sx"] = {"data": sx.tolist(), "operation": "sum", "type": "float"}
    transfer_["sxx"] = {"data": sxx.tolist(), "operation": "sum", "type": "float"}
    transfer_["sxy"] = {"data": sxy.tolist(), "operation": "sum", "type": "float"}
    transfer_["sy"] = {"data": sy.tolist(), "operation": "sum", "type": "float"}
    transfer_["syy"] = {"data": syy.tolist(), "operation": "sum", "type": "float"}

    return transfer_


@udf(
    local_transfers=secure_transfer(sum_op=True),
    alpha=literal(),
    return_type=[transfer()],
)
def global1(local_transfers, alpha):
    import scipy.special as special
    import scipy.stats as st

    n_obs = local_transfers["n_obs"]
    sx = numpy.array(local_transfers["sx"])
    sy = numpy.array(local_transfers["sy"])
    sxx = numpy.array(local_transfers["sxx"])
    sxy = numpy.array(local_transfers["sxy"])
    syy = numpy.array(local_transfers["syy"])

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
    z = st.norm.ppf(1 - alpha / 2)
    lo_z, hi_z = r_z - z * se, r_z + z * se
    ci_lo, ci_hi = numpy.tanh((lo_z, hi_z))

    transfer_ = {
        "n_obs": n_obs,
        "correlations": correlations.tolist(),
        "p_values": p_values.tolist(),
        "ci_lo": ci_lo.tolist(),
        "ci_hi": ci_hi.tolist(),
    }

    return transfer_


def get_var_pair_names(x_names, y_names):
    tildas = numpy.empty((len(y_names), len(x_names)), dtype=object)
    tildas[:] = " ~ "
    pair_names = y_names[:, numpy.newaxis] + tildas + x_names
    return pair_names

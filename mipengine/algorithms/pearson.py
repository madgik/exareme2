import json
import numpy

from typing import TypeVar

from pydantic import BaseModel
from mipengine.udfgen.udfgenerator import (
    udf,
    transfer,
    relation,
    make_unique_func_name,
    merge_transfer,
    literal,
)


class PearsonResult(BaseModel):
    title: str
    n_obs: int
    correlations: dict
    p_values: dict
    ci_hi: dict
    ci_lo: dict


def run(algo_interface):
    local_run = algo_interface.run_udf_on_local_nodes
    global_run = algo_interface.run_udf_on_global_node
    Y_relation = algo_interface.initial_view_tables["y"]
    alpha = algo_interface.algorithm_parameters["alpha"]

    if "x" in algo_interface.initial_view_tables:
        X_relation = algo_interface.initial_view_tables["x"]
    else:
        X_relation = algo_interface.initial_view_tables["y"]

    column_names = [
        x.__dict__["name"]
        for x in Y_relation.get_table_schema().__dict__["columns"]
        if x.__dict__["name"] != "row_id"
    ]

    row_names = [
        x.__dict__["name"]
        for x in X_relation.get_table_schema().__dict__["columns"]
        if x.__dict__["name"] != "row_id"
    ]

    local_transfers = local_run(
        func_name=make_unique_func_name(local1),
        keyword_args=dict(y=Y_relation, x=X_relation),
        share_to_global=[True],
    )

    result = global_run(
        func_name=make_unique_func_name(global1),
        keyword_args=dict(local_transfers=local_transfers, alpha=alpha),
    )

    result = json.loads(result.get_table_data()[1][0])
    n_obs = result["n_obs"]

    correlations = result["correlations"]
    p_values = result["p_values"]
    ci_hi = result["ci_hi"]
    ci_lo = result["ci_lo"]

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

    result = PearsonResult(
        title="Pearson Correlation Coefficient",
        n_obs=n_obs,
        correlations=corr_dict,
        p_values=p_values_dict,
        ci_hi=ci_hi_dict,
        ci_lo=ci_lo_dict,
    )
    return result


S = TypeVar("S")


@udf(y=relation(schema=S), x=relation(schema=S), return_type=[transfer()])
def local1(y, x):
    n_obs = y.shape[0]
    Y = y.to_numpy()
    X = Y if x is None else x.to_numpy()

    sx = X.sum(axis=0)
    sy = Y.sum(axis=0)
    sxx = (X ** 2).sum(axis=0)
    sxy = (X * Y.T[:, :, numpy.newaxis]).sum(axis=1)
    syy = (Y ** 2).sum(axis=0)

    transfer_ = {}
    transfer_["n_obs"] = n_obs
    transfer_["sx"] = sx.tolist()
    transfer_["sxx"] = sxx.tolist()
    transfer_["sxy"] = sxy.tolist()
    transfer_["sy"] = sy.tolist()
    transfer_["syy"] = syy.tolist()

    return transfer_


@udf(local_transfers=merge_transfer(), alpha=literal(), return_type=[transfer()])
def global1(local_transfers, alpha):
    import scipy.special as special
    import scipy.stats as st

    P_VALUE_CUTOFF = 0.001
    # P_VALUE_CUTOFF_STR = "< " + str(P_VALUE_CUTOFF)
    CONFIDENCE = 0.95

    n_obs = sum(t["n_obs"] for t in local_transfers)
    sx = sum(numpy.array(t["sx"]) for t in local_transfers)
    sy = sum(numpy.array(t["sy"]) for t in local_transfers)
    sxx = sum(numpy.array(t["sxx"]) for t in local_transfers)
    sxy = sum(numpy.array(t["sxy"]) for t in local_transfers)
    syy = sum(numpy.array(t["syy"]) for t in local_transfers)

    df = n_obs - 2
    d = (
        numpy.sqrt(n_obs * sxx - sx * sx)
        * numpy.sqrt(n_obs * syy - sy * sy)[:, numpy.newaxis]
    )
    correlations = (n_obs * sxy - sx * sy[:, numpy.newaxis]) / d
    correlations[d == 0] = 0
    correlations = correlations.clip(-1, 1)
    t_squared = correlations ** 2 * (df / ((1.0 - correlations) * (1.0 + correlations)))
    p_values = special.betainc(
        0.5 * df, 0.5, numpy.fmin(numpy.asarray(df / (df + t_squared)), 1.0)
    )
    p_values[abs(correlations) == 1] = 0
    r_z = numpy.arctanh(correlations)
    se = 1 / numpy.sqrt(n_obs - 3)
    # alpha = 1 - CONFIDENCE
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

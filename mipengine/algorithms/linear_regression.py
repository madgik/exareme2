import json
from typing import List
from typing import TypeVar

import numpy
import scipy.stats as stats
from pydantic import BaseModel

from mipengine.udfgen import literal
from mipengine.udfgen import relation
from mipengine.udfgen import secure_transfer
from mipengine.udfgen import state
from mipengine.udfgen import transfer
from mipengine.udfgen import udf

ALPHA = 0.05  # NOTE maybe this should be a model parameter


class LinearRegressionResult(BaseModel):
    dependent_var: str
    n_obs: int
    df_resid: float
    df_model: float
    rse: float
    r_squared: float
    r_squared_adjusted: float
    f_stat: float
    f_pvalue: float
    indep_vars: List[str]
    coefficients: List[float]
    std_err: List[float]
    t_stats: List[float]
    pvalues: List[float]
    lower_ci: List[float]
    upper_ci: List[float]


def run(algo_interface):
    x, y = algo_interface.create_primary_data_views(
        variable_groups=[algo_interface.x_variables, algo_interface.y_variables],
    )
    p = len(algo_interface.x_variables)

    lr = LinearRegression(algo_interface)
    lr.fit(X=x, y=y)
    lr.predict(x)  # saves y_pred in local_states
    lr.compute_summary(y=y, y_mean=lr.y_mean_, p=p)

    result = LinearRegressionResult(
        dependent_var=algo_interface.y_variables[0],
        n_obs=lr.n_obs_,
        df_resid=lr.df_,
        df_model=p,
        rse=lr.rse_,
        r_squared=lr.r_squared_,
        r_squared_adjusted=lr.r_squared_adjusted_,
        f_stat=lr.f_stat_,
        f_pvalue=lr.f_p_value_,
        indep_vars=algo_interface.x_variables,
        coefficients=[c[0] for c in lr.coefficients_],
        std_err=lr.std_err_.tolist(),
        t_stats=lr.t_stat_.tolist(),
        pvalues=lr.t_p_values_.tolist(),
        lower_ci=lr.ci_[0].tolist(),
        upper_ci=lr.ci_[1].tolist(),
    )
    return result


class LinearRegression:
    def __init__(self, algo_interface):
        self.local_run = algo_interface.run_udf_on_local_nodes
        self.global_run = algo_interface.run_udf_on_global_node

    def fit(self, X, y) -> numpy.ndarray:
        local_transfers = self.local_run(
            func=fit_local,
            keyword_args={"x": X, "y": y},
            share_to_global=[True],
        )
        self.global_state, self.global_transfer = self.global_run(
            func=fit_global,
            keyword_args=dict(local_transfers=local_transfers),
            share_to_locals=[False, True],
        )
        global_transfer_data = json.loads(self.global_transfer.get_table_data()[1][0])
        self.coefficients_ = global_transfer_data["coefficients"]
        self.y_mean_ = global_transfer_data["y_mean"]

    def predict(self, X):
        self.local_states = self.local_run(
            func=predict_local,
            keyword_args=dict(x=X, coefficients=self.coefficients_),
            share_to_global=[False],
        )

    def compute_summary(self, y, y_mean, p):
        local_transfers = self.local_run(
            func=compute_summary_local,
            keyword_args=dict(y=y, y_mean=y_mean, predict_states=self.local_states),
            share_to_global=[True],
        )
        global_transfer = self.global_run(
            func=compute_summary_global,
            keyword_args=dict(
                local_transfers=local_transfers, fit_gstate=self.global_state
            ),
        )
        global_transfer_data = json.loads(global_transfer.get_table_data()[1][0])
        rss = global_transfer_data["rss"]
        tss = global_transfer_data["tss"]
        xTx_inv = numpy.array(global_transfer_data["xTx_inv"])
        coefficients = numpy.array(self.coefficients_)
        n_obs = global_transfer_data["n_obs"]
        df = n_obs - p - 1
        self.n_obs_ = n_obs
        self.df_ = df
        self.rse_ = (rss / df) ** 0.5
        self.std_err_ = ((self.rse_**2) * numpy.diag(xTx_inv)) ** 0.5
        self.t_stat_ = coefficients.T[0] / self.std_err_
        self.ci_ = (
            coefficients.T[0] - stats.t.ppf(1 - ALPHA / 2, df) * self.std_err_,
            coefficients.T[0] + stats.t.ppf(1 - ALPHA / 2, df) * self.std_err_,
        )
        self.r_squared_ = 1.0 - rss / tss
        self.r_squared_adjusted_ = 1 - (1 - self.r_squared_) * (n_obs - 1) / df
        self.f_stat_ = (tss - rss) * df / (p * rss)
        self.t_p_values_ = stats.t.sf(abs(self.t_stat_), df=df) * 2
        self.f_p_value_ = stats.f.sf(self.f_stat_, dfn=p, dfd=df)


S1 = TypeVar("S1")
S2 = TypeVar("S2")


@udf(x=relation(S1), y=relation(S2), return_type=[secure_transfer(sum_op=True)])
def fit_local(x, y):
    x.insert(0, "Intercept", [1] * len(x))  # TODO move to preprocessing function
    xTx = x.T @ x
    xTy = x.T @ y
    sy = float(y.sum())
    n_obs = len(y)
    transfer_ = {}
    transfer_["xTx"] = {"data": xTx.to_numpy().tolist(), "operation": "sum"}
    transfer_["xTy"] = {"data": xTy.to_numpy().tolist(), "operation": "sum"}
    transfer_["sy"] = {"data": sy, "operation": "sum"}
    transfer_["n_obs"] = {"data": n_obs, "operation": "sum"}
    return transfer_


@udf(local_transfers=secure_transfer(sum_op=True), return_type=[state(), transfer()])
def fit_global(local_transfers):
    xTx = numpy.array(local_transfers["xTx"])
    xTy = numpy.array(local_transfers["xTy"])
    sy = local_transfers["sy"]
    n_obs = local_transfers["n_obs"]

    xTx_inv = numpy.linalg.pinv(xTx)
    coefficients = xTx_inv @ xTy

    y_mean = sy / n_obs

    state_ = {}
    state_["xTx_inv"] = xTx_inv  # Needed for SE(β) calculation
    state_["n_obs"] = n_obs

    transfer_ = {}
    transfer_["coefficients"] = coefficients.tolist()
    transfer_["y_mean"] = y_mean
    return state_, transfer_


@udf(x=relation(S1), coefficients=literal(), return_type=state())
def predict_local(x, coefficients):
    x.insert(0, "Intercept", [1] * len(x))  # TODO move to preprocessing function
    coefficients = numpy.array(coefficients)

    y_pred = x @ coefficients

    state_ = {}
    state_["y_pred"] = y_pred
    return state_


@udf(
    y=relation(S1),
    y_mean=literal(),
    predict_states=state(),
    return_type=secure_transfer(sum_op=True),
)
def compute_summary_local(y, y_mean, predict_states):
    y_pred = predict_states["y_pred"]
    rss = float(sum((y.to_numpy() - y_pred.to_numpy()) ** 2))
    tss = float(sum((y.to_numpy() - y_mean) ** 2))

    transfer_ = {}
    transfer_["rss"] = {"data": rss, "operation": "sum"}
    transfer_["tss"] = {"data": tss, "operation": "sum"}
    return transfer_


@udf(
    fit_gstate=state(),
    local_transfers=secure_transfer(sum_op=True),
    return_type=transfer(),
)
def compute_summary_global(fit_gstate, local_transfers):
    xTx_inv = fit_gstate["xTx_inv"]
    n_obs = fit_gstate["n_obs"]
    rss = local_transfers["rss"]
    tss = local_transfers["tss"]

    transfer_ = {}
    transfer_["rss"] = rss
    transfer_["tss"] = tss
    transfer_["n_obs"] = n_obs
    transfer_["xTx_inv"] = xTx_inv.tolist()
    return transfer_


@udf(y=relation(S1), prev_state=state(), return_type=transfer())
def compute_mse_local(y, prev_state):
    y_pred = prev_state["y_pred"]
    rss = float(sum((y.to_numpy() - y_pred.to_numpy()) ** 2))
    n_obs = len(y)

    transfer_ = {}
    transfer_["rss"] = rss
    transfer_["n_obs"] = n_obs
    return transfer_


@udf(local_transfers=transfer(), return_type=relation(schema=[("scalar", float)]))
def compute_mse_global(local_transfers):
    rss = local_transfers["rss"]
    n_obs = local_transfers["n_obs"]
    mse = rss / n_obs
    return mse

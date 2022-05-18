import json
from typing import List

import numpy
import scipy.stats as stats
from pydantic import BaseModel

from mipengine.algorithms.preprocessing import DummyEncoder
from mipengine.algorithms.preprocessing import relation_to_vector
from mipengine.udfgen import literal
from mipengine.udfgen import relation
from mipengine.udfgen import secure_transfer
from mipengine.udfgen import state
from mipengine.udfgen import tensor
from mipengine.udfgen import transfer
from mipengine.udfgen import udf

ALPHA = 0.05  # NOTE maybe this should be a model parameter

RealVector = tensor(dtype=float, ndims=1)


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


def run(executor):
    X, y = executor.create_primary_data_views(
        variable_groups=[executor.x_variables, executor.y_variables],
    )

    dummy_encoder = DummyEncoder(executor)
    X = dummy_encoder.transform(X)

    p = len(dummy_encoder.new_varnames) - 1

    lr = LinearRegression(executor)
    lr.fit(X=X, y=y)
    y_pred: RealVector = lr.predict(X)
    lr.compute_summary(
        y=relation_to_vector(y, executor),
        y_pred=y_pred,
        y_mean=lr.y_mean,
        p=p,
    )

    result = LinearRegressionResult(
        dependent_var=executor.y_variables[0],
        n_obs=lr.n_obs,
        df_resid=lr.df,
        df_model=p,
        rse=lr.rse,
        r_squared=lr.r_squared,
        r_squared_adjusted=lr.r_squared_adjusted,
        f_stat=lr.f_stat,
        f_pvalue=lr.f_p_value,
        indep_vars=dummy_encoder.new_varnames,
        coefficients=[c[0] for c in lr.coefficients],
        std_err=lr.std_err.tolist(),
        t_stats=lr.t_stat.tolist(),
        pvalues=lr.t_p_values.tolist(),
        lower_ci=lr.ci[0].tolist(),
        upper_ci=lr.ci[1].tolist(),
    )
    return result


class LinearRegression:
    def __init__(self, executor):
        self.local_run = executor.run_udf_on_local_nodes
        self.global_run = executor.run_udf_on_global_node

    def fit(self, X, y):
        local_transfers = self.local_run(
            func=self._fit_local,
            keyword_args={"x": X, "y": y},
            share_to_global=[True],
        )
        self.global_state, self.global_transfer = self.global_run(
            func=self._fit_global,
            keyword_args=dict(local_transfers=local_transfers),
            share_to_locals=[False, True],
        )
        global_transfer_data = json.loads(self.global_transfer.get_table_data()[1][0])
        self.coefficients = global_transfer_data["coefficients"]
        self.y_mean = global_transfer_data["y_mean"]

    @staticmethod
    @udf(x=relation(), y=relation(), return_type=[secure_transfer(sum_op=True)])
    def _fit_local(x, y):
        xTx = x.T @ x
        xTy = x.T @ y
        sy = float(y.sum())
        n_obs = len(y)
        stransfer = {}
        stransfer["xTx"] = {"data": xTx.to_numpy().tolist(), "operation": "sum"}
        stransfer["xTy"] = {"data": xTy.to_numpy().tolist(), "operation": "sum"}
        stransfer["sy"] = {"data": sy, "operation": "sum"}
        stransfer["n_obs"] = {"data": n_obs, "operation": "sum"}
        return stransfer

    @staticmethod
    @udf(
        local_transfers=secure_transfer(sum_op=True),
        return_type=[state(), transfer()],
    )
    def _fit_global(local_transfers):
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

    def predict(self, X):
        return self.local_run(
            func=self._predict_local,
            keyword_args=dict(x=X, coefficients=self.coefficients),
            share_to_global=[False],
        )

    @staticmethod
    @udf(x=relation(), coefficients=literal(), return_type=RealVector)
    def _predict_local(x, coefficients):
        x = x.values
        coefficients = numpy.array(coefficients)

        y_pred = x @ coefficients
        return y_pred

    def compute_summary(self, y, y_pred, y_mean, p):
        local_transfers = self.local_run(
            func=self._compute_summary_local,
            keyword_args=dict(y=y, y_pred=y_pred, y_mean=y_mean),
            share_to_global=[True],
        )
        global_transfer = self.global_run(
            func=self._compute_summary_global,
            keyword_args=dict(
                local_transfers=local_transfers, fit_gstate=self.global_state
            ),
        )
        global_transfer_data = json.loads(global_transfer.get_table_data()[1][0])
        rss = global_transfer_data["rss"]
        tss = global_transfer_data["tss"]
        xTx_inv = numpy.array(global_transfer_data["xTx_inv"])
        coefficients = numpy.array(self.coefficients)
        n_obs = global_transfer_data["n_obs"]
        df = n_obs - p - 1
        self.n_obs = n_obs
        self.df = df
        self.rse = (rss / df) ** 0.5
        self.std_err = ((self.rse**2) * numpy.diag(xTx_inv)) ** 0.5
        self.t_stat = coefficients.T[0] / self.std_err
        self.ci = (
            coefficients.T[0] - stats.t.ppf(1 - ALPHA / 2, df) * self.std_err,
            coefficients.T[0] + stats.t.ppf(1 - ALPHA / 2, df) * self.std_err,
        )
        self.r_squared = 1.0 - rss / tss
        self.r_squared_adjusted = 1 - (1 - self.r_squared) * (n_obs - 1) / df
        self.f_stat = (tss - rss) * df / (p * rss)
        self.t_p_values = stats.t.sf(abs(self.t_stat), df=df) * 2
        self.f_p_value = stats.f.sf(self.f_stat, dfn=p, dfd=df)

    @staticmethod
    @udf(
        y=RealVector,
        y_pred=RealVector,
        y_mean=literal(),
        return_type=secure_transfer(sum_op=True),
    )
    def _compute_summary_local(y, y_pred, y_mean):
        rss = float(sum((y - y_pred) ** 2))
        tss = float(sum((y - y_mean) ** 2))

        stransfer = {}
        stransfer["rss"] = {"data": rss, "operation": "sum"}
        stransfer["tss"] = {"data": tss, "operation": "sum"}
        return stransfer

    @staticmethod
    @udf(
        fit_gstate=state(),
        local_transfers=secure_transfer(sum_op=True),
        return_type=transfer(),
    )
    def _compute_summary_global(fit_gstate, local_transfers):
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

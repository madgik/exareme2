from typing import List

import numpy
import scipy.stats as stats
from pydantic import BaseModel
from scipy.special import xlogy

from mipengine.algorithms.algorithm import Algorithm
from mipengine.algorithms.helpers import get_transfer_data
from mipengine.algorithms.preprocessing import DummyEncoder
from mipengine.algorithms.preprocessing import LabelBinarizer
from mipengine.exceptions import BadUserInput
from mipengine.udfgen import literal
from mipengine.udfgen import relation
from mipengine.udfgen import secure_transfer
from mipengine.udfgen import transfer
from mipengine.udfgen import udf

MAX_ITER = 50  # maximum iterations before cancelling run due to non convergence
TOL = 1e-4  # tolerance for stopping criterion
ALPHA = 0.05  # alpha level for coefficient confidence intervals


class LogisticRegressionAlgorithm(Algorithm, algname="logistic_regression"):
    def get_variable_groups(self):
        return [self.variables.x, self.variables.y]

    def run(self, executor):
        X, y = executor.data_model_views
        positive_class = self.algorithm_parameters["positive_class"]

        dummy_encoder = DummyEncoder(
            executor=executor, variables=self.variables, metadata=self.metadata
        )
        X = dummy_encoder.transform(X)

        ybin = LabelBinarizer(executor, positive_class).transform(y)

        lr = LogisticRegression(executor)
        lr.fit(X=X, y=ybin)

        summary = compute_summary(model=lr)

        result = LogisticRegressionResult(
            dependent_var=y.columns[0],
            indep_vars=X.columns,
            summary=summary,
        )

        return result


class LogisticRegression:
    def __init__(self, executor):
        self.local_run = executor.run_udf_on_local_nodes
        self.global_run = executor.run_udf_on_global_node

    def fit(self, X, y):
        self.p = len(X.columns)

        # init model
        coeff = [0] * self.p
        local_transfers = self.local_run(
            self._fit_init_local,
            keyword_args={"y": y},
            share_to_global=True,
        )
        global_transfer = self.global_run(
            self._fit_init_global,
            keyword_args={"local_transfers": local_transfers},
        )
        transfer_data = get_transfer_data(global_transfer)
        self.nobs_train = transfer_data["nobs_train"]
        self.y_sum = transfer_data["y_sum"]
        handle_logreg_errors(self.nobs_train, self.p, self.y_sum)

        # optimization iteration
        for i in range(MAX_ITER):
            local_transfers = self.local_run(
                self._fit_local_step,
                keyword_args={"X": X, "y": y, "coeff": coeff},
                share_to_global=True,
            )
            global_transfer = self.global_run(
                self._fit_global_step,
                keyword_args={"local_transfers": local_transfers, "coeff": coeff},
            )
            transfer_data = get_transfer_data(global_transfer)
            coeff = transfer_data["coeff"]
            grad = transfer_data["grad"]

            # stopping criterion
            if max_abs(grad) <= TOL:
                break
        else:
            raise BadUserInput("Logistic regression cannot converge. Cancelling run.")

        self.coeff = coeff
        self.ll = transfer_data["ll"]
        self.H_inv = transfer_data["H_inv"]

    @staticmethod
    @udf(y=relation(), return_type=secure_transfer(sum_op=True))
    def _fit_init_local(y):
        nobs_train = len(y)
        y_sum = y.sum()
        stransfer = {}
        stransfer["nobs_train"] = {
            "data": nobs_train,
            "operation": "sum",
            "type": "int",
        }
        stransfer["y_sum"] = {
            "data": int(y_sum),
            "operation": "sum",
            "type": "int",
        }
        return stransfer

    @staticmethod
    @udf(local_transfers=secure_transfer(sum_op=True), return_type=transfer())
    def _fit_init_global(local_transfers):
        nobs_train = local_transfers["nobs_train"]
        y_sum = local_transfers["y_sum"]
        transfer_ = {"nobs_train": nobs_train, "y_sum": y_sum}
        return transfer_

    @staticmethod
    @udf(
        X=relation(),
        y=relation(),
        coeff=literal(),
        return_type=secure_transfer(sum_op=True),
    )
    def _fit_local_step(X, y, coeff):
        from scipy import special

        # Add a second axis to coeff to make it a proper column vector.
        # Simplifies algebraic manipulations later.
        coeff = numpy.array(coeff)[:, numpy.newaxis]

        X = X.to_numpy()
        y = y.to_numpy()

        # auxiliary quantities
        eta = X @ coeff
        mu = special.expit(eta)
        w = mu * (1 - mu)

        # The computation of the Hessian could have been writen as
        #     X.T @ numpy.diag(d) @ X
        # However, this generates a large (n_obs, n_obs) diagonal matrix.
        # Instead, the version using Einstein summation is memory efficient
        # thanks to the optimized tensor constraction algorithms behind einsum.
        H = numpy.einsum("ji, j..., jk -> ik", X, w, X, optimize="greedy")

        # gradient
        grad = numpy.einsum("ji, j... -> i", X, y - mu, optimize="greedy")

        # log-likelihood
        ll = numpy.sum(special.xlogy(y, mu) + special.xlogy(1 - y, 1 - mu))

        stransfer = {}
        stransfer["H"] = {"data": H.tolist(), "operation": "sum", "type": "float"}
        stransfer["grad"] = {"data": grad.tolist(), "operation": "sum", "type": "float"}
        stransfer["ll"] = {"data": float(ll), "operation": "sum", "type": "float"}
        return stransfer

    @staticmethod
    @udf(
        local_transfers=secure_transfer(sum_op=True),
        coeff=literal(),
        return_type=transfer(),
    )
    def _fit_global_step(local_transfers, coeff):
        ll = local_transfers["ll"]
        ll = numpy.array(ll)
        grad = local_transfers["grad"]
        grad = numpy.array(grad)
        H = local_transfers["H"]
        H = numpy.array(H)

        # if inverse fails try Moore-Penrose pseudo-inverse
        try:
            H_inv = numpy.linalg.inv(H)
        except numpy.linalg.LinAlgError:
            H_inv = numpy.linalg.pinv(H)

        # update coeff according to Newton's method
        coeff += H_inv @ grad

        transfer_ = {}
        transfer_["ll"] = float(ll)
        transfer_["coeff"] = coeff.tolist()
        transfer_["grad"] = grad.tolist()
        transfer_["H_inv"] = H_inv.tolist()
        return transfer_

    def predict_proba(self, X):
        return self.local_run(
            self._predict_proba_local, keyword_args={"X": X, "coeff": self.coeff}
        )

    @staticmethod
    @udf(
        X=relation(),
        coeff=literal(),
        return_type=relation(schema=[("row_id", int), ("proba", float)]),
    )
    def _predict_proba_local(X, coeff):
        import pandas as pd
        from scipy import special

        index = X.index
        X = X.to_numpy()
        coeff = numpy.array(coeff)[:, numpy.newaxis]

        proba = special.expit(X @ coeff)

        result = pd.DataFrame({"row_id": index, "proba": numpy.squeeze(proba)})
        return result


def handle_logreg_errors(nobs, p, y_sum):
    if nobs <= p:
        msg = (
            "Logistic regression cannot run because the number of "
            "observarions is smaller than the number of predictors. Please "
            "add mode predictors or select more observations."
        )
        raise BadUserInput(msg)
    if min(y_sum, nobs - y_sum) <= p:
        msg = (
            "Logistic regression cannot run because the number of "
            "observarions in one category is smaller than the number of "
            "predictors. Please add mode predictors or select more "
            "observations for the category in question."
        )
        raise BadUserInput(msg)


def compute_summary(model: LogisticRegression):
    p = model.p
    H_inv = model.H_inv
    coeff = model.coeff
    nobs_train = model.nobs_train
    y_sum = model.y_sum
    ll = model.ll

    # statistics
    stderr = numpy.sqrt(numpy.diag(H_inv))
    z_scores = coeff / stderr
    pvalues = stats.norm.sf(abs(z_scores)) * 2

    # confidence intervals
    lower_ci = [c - s * stats.norm.ppf(1 - ALPHA / 2) for c, s in zip(coeff, stderr)]
    upper_ci = [c + s * stats.norm.ppf(1 - ALPHA / 2) for c, s in zip(coeff, stderr)]

    # degrees of freedom
    df_model = p - 1
    df_resid = nobs_train - p

    # Null model log-likelihood
    y_mean = y_sum / nobs_train
    ll0 = xlogy(y_sum, y_mean) + xlogy(nobs_train - y_sum, 1.0 - y_mean)

    # AIC
    aic = 2 * p - 2 * ll

    # BIC
    bic = numpy.log(nobs_train) * p - 2 * ll

    # pseudo-R^2 McFadden and Cox-Snell
    if numpy.isclose(ll, 0.0) and numpy.isclose(ll0, 0.0):
        r2_mcf = 1
    else:
        r2_mcf = 1 - ll / ll0
    r2_cs = 1 - numpy.exp(2 * (ll0 - ll) / nobs_train)

    return LogisticRegressionSummary(
        n_obs=nobs_train,
        coefficients=coeff,
        stderr=stderr.tolist(),
        lower_ci=lower_ci,
        upper_ci=upper_ci,
        z_scores=z_scores.tolist(),
        pvalues=pvalues.tolist(),
        df_resid=df_resid,
        df_model=df_model,
        r_squared_cs=r2_cs,
        r_squared_mcf=r2_mcf,
        aic=aic,
        bic=bic,
        ll0=ll0,
        ll=ll,
    )


class LogisticRegressionSummary(BaseModel):
    n_obs: int
    coefficients: List[float]
    stderr: List[float]
    lower_ci: List[float]
    upper_ci: List[float]
    z_scores: List[float]
    pvalues: List[float]
    df_model: int
    df_resid: int
    r_squared_cs: float
    r_squared_mcf: float
    ll0: float
    ll: float
    aic: float
    bic: float


class LogisticRegressionResult(BaseModel):
    dependent_var: str
    indep_vars: List[str]
    summary: LogisticRegressionSummary


def max_abs(lst):
    return max(abs(elm) for elm in lst)

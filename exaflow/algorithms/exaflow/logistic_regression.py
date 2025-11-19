from typing import List

import numpy as np
from pydantic import BaseModel
from scipy import stats
from scipy.special import expit
from scipy.special import xlogy

from exaflow.algorithms.exaflow.algorithm import Algorithm
from exaflow.algorithms.exaflow.exaflow_registry import exaflow_udf
from exaflow.algorithms.exaflow.metrics import build_design_matrix
from exaflow.algorithms.exaflow.metrics import collect_categorical_levels_from_df
from exaflow.algorithms.exaflow.metrics import construct_design_labels
from exaflow.algorithms.exaflow.metrics import get_dummy_categories
from exaflow.worker_communication import BadUserInput

ALGORITHM_NAME = "logistic_regression"
MAX_ITER = 50
TOL = 1e-4
ALPHA = 0.05


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


class LogisticRegressionAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self, metadata: dict):
        if not self.inputdata.y:
            raise BadUserInput("Logistic regression requires a dependent variable.")
        if not self.inputdata.x:
            raise BadUserInput("Logistic regression requires at least one covariate.")

        positive_class = self.parameters.get("positive_class")
        if positive_class is None:
            raise BadUserInput("Parameter 'positive_class' is required.")

        y_var = self.inputdata.y[0]
        categorical_vars = [
            var for var in self.inputdata.x if metadata[var]["is_categorical"]
        ]
        numerical_vars = [
            var for var in self.inputdata.x if not metadata[var]["is_categorical"]
        ]

        # ✅ Discover dummies from actual data (not metadata)
        dummy_categories = get_dummy_categories(
            self.engine,
            self.inputdata.json(),
            categorical_vars,
            logistic_collect_categorical_levels,
        )

        indep_var_names = construct_design_labels(
            categorical_vars, dummy_categories, numerical_vars
        )

        udf_results = self.engine.run_algorithm_udf(
            func=logistic_regression_local_step,
            positional_args={
                "inputdata": self.inputdata.json(),
                "positive_class": positive_class,
                "y_var": y_var,
                "categorical_vars": categorical_vars,
                "numerical_vars": numerical_vars,
                "dummy_categories": dummy_categories,
            },
        )

        model_stats = udf_results[0]
        summary = compute_summary(
            coefficients=np.array(model_stats["coefficients"], dtype=float),
            h_inv=np.array(model_stats["hessian_inverse"], dtype=float),
            ll=model_stats["ll"],
            n_obs=model_stats["n_obs"],
            y_sum=model_stats["y_sum"],
        )

        return LogisticRegressionResult(
            dependent_var=y_var,
            indep_vars=indep_var_names,
            summary=summary,
        )


@exaflow_udf()
def logistic_collect_categorical_levels(data, inputdata, categorical_vars):

    return collect_categorical_levels_from_df(data, categorical_vars)


@exaflow_udf(with_aggregation_server=True)
def logistic_regression_local_step(
    data,
    inputdata,
    agg_client,
    positive_class,
    y_var,
    categorical_vars,
    numerical_vars,
    dummy_categories,
):
    # --- keep only the variables we actually use (X + y) and ensure unique names ---
    # order: categorical, numerical, then y
    cols = list(dict.fromkeys(list(categorical_vars) + list(numerical_vars) + [y_var]))

    # subset and drop duplicated column names, keeping the first occurrence
    data = data[cols].copy()
    if data.empty:
        X = build_design_matrix(
            data,
            categorical_vars=categorical_vars,
            dummy_categories=dummy_categories,
            numerical_vars=numerical_vars,
        )
        y = np.empty((0, 1), dtype=float)
    else:
        # y_var is now guaranteed to be a single 1D column, not a 2D frame
        y = (data[y_var] == positive_class).astype(float).to_numpy().reshape(-1, 1)

        X = build_design_matrix(
            data,
            categorical_vars=categorical_vars,
            dummy_categories=dummy_categories,
            numerical_vars=numerical_vars,
        )

    model_stats = run_distributed_logistic_regression(agg_client, X, y)
    return model_stats


def run_distributed_logistic_regression(agg_client, X: np.ndarray, y: np.ndarray):
    if X.ndim != 2:
        X = np.atleast_2d(X)
    if y.ndim == 2 and y.shape[1] != 1:
        y = y.reshape(-1, 1)
    if X.shape[0] != y.shape[0]:
        if X.shape[1] == y.shape[0]:
            X = X.T
        else:
            raise BadUserInput(
                "Design matrix row count does not match target size for logistic regression."
            )

    n_obs_local = int(y.size)
    y_sum_local = float(y.sum())

    total_n_obs = int(agg_client.sum([float(n_obs_local)])[0])
    total_y_sum = float(agg_client.sum([float(y_sum_local)])[0])

    n_features = X.shape[1]
    handle_logreg_errors(total_n_obs, n_features, total_y_sum)

    coeff = np.zeros((n_features, 1), dtype=float)
    H_inv = np.eye(n_features, dtype=float)
    ll = 0.0

    for _ in range(MAX_ITER):
        eta = X @ coeff
        mu = expit(eta)
        w = mu * (1.0 - mu)

        grad_local = np.einsum("ji,j->i", X, (y - mu).reshape(-1))
        H_local = np.einsum("ji,j,jk->ik", X, w.reshape(-1), X)
        ll_local = np.sum(xlogy(y, mu) + xlogy(1 - y, 1 - mu))

        grad = np.asarray(agg_client.sum(grad_local.tolist()), dtype=float)
        H_flat = agg_client.sum(H_local.ravel().tolist())
        H = np.asarray(H_flat, dtype=float).reshape((n_features, n_features))
        ll = float(agg_client.sum([float(ll_local)])[0])

        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            H_inv = np.linalg.pinv(H)

        coeff = coeff + H_inv @ grad.reshape(-1, 1)

        if max_abs(grad) <= TOL:
            break
    else:
        raise BadUserInput("Logistic regression cannot converge. Cancelling run.")

    return {
        "coefficients": coeff.reshape(-1).tolist(),
        "hessian_inverse": H_inv.tolist(),
        "ll": ll,
        "n_obs": total_n_obs,
        "y_sum": total_y_sum,
    }


def compute_summary(
    *,
    coefficients: np.ndarray,
    h_inv: np.ndarray,
    ll: float,
    n_obs: int,
    y_sum: float,
) -> LogisticRegressionSummary:
    stderr = np.sqrt(np.diag(h_inv))
    z_scores = np.divide(
        coefficients, stderr, out=np.zeros_like(coefficients), where=stderr != 0
    )
    pvalues = stats.norm.sf(np.abs(z_scores)) * 2

    z_crit = stats.norm.ppf(1 - ALPHA / 2)
    lower_ci = (coefficients - z_crit * stderr).tolist()
    upper_ci = (coefficients + z_crit * stderr).tolist()

    df_model = len(coefficients) - 1
    df_resid = n_obs - len(coefficients)

    y_mean = y_sum / n_obs if n_obs else 0
    ll0 = float(xlogy(y_sum, y_mean) + xlogy(n_obs - y_sum, 1.0 - y_mean))

    aic = 2 * len(coefficients) - 2 * ll
    bic = np.log(n_obs) * len(coefficients) - 2 * ll if n_obs else float("inf")

    if np.isclose(ll, 0.0) and np.isclose(ll0, 0.0):
        r2_mcf = 1.0
    else:
        r2_mcf = 1.0 - (ll / ll0)
    r2_cs = 1.0 - np.exp(2.0 * (ll0 - ll) / n_obs) if n_obs else 0.0

    return LogisticRegressionSummary(
        n_obs=int(n_obs),
        coefficients=coefficients.tolist(),
        stderr=stderr.tolist(),
        lower_ci=lower_ci,
        upper_ci=upper_ci,
        z_scores=z_scores.tolist(),
        pvalues=pvalues.tolist(),
        df_model=int(df_model),
        df_resid=int(df_resid),
        r_squared_cs=r2_cs,
        r_squared_mcf=r2_mcf,
        ll0=ll0,
        ll=float(ll),
        aic=aic,
        bic=bic,
    )


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


def max_abs(values):
    return float(np.max(np.abs(values))) if len(values) else 0.0

from typing import Dict

import numpy as np
from scipy import stats
from scipy.special import expit
from scipy.special import xlogy

from exaflow.algorithms.exareme3.library.lazy_aggregation import lazy_agg
from exaflow.worker_communication import BadUserInput

MAX_ITER = 50
TOL = 1e-4


def handle_logreg_errors(nobs: int, p: int, y_sum: float):
    if nobs <= p:
        msg = (
            "Logistic regression cannot run because the number of "
            "observations is smaller than the number of predictors. Please "
            "add more predictors or select more observations."
        )
        raise BadUserInput(msg)
    if min(y_sum, nobs - y_sum) <= p:
        msg = (
            "Logistic regression cannot run because the number of "
            "observations in one category is smaller than the number of "
            "predictors. Please add more predictors or select more "
            "observations for the category in question."
        )
        raise BadUserInput(msg)


def max_abs(values: np.ndarray) -> float:
    return float(np.max(np.abs(values))) if len(values) else 0.0


@lazy_agg()
def run_distributed_logistic_regression(
    agg_client, X: np.ndarray, y: np.ndarray
) -> Dict:
    """
    Newton–Raphson fit of a logistic model with secure aggregation.

    Mathematical notes (per iteration):
    - Linear predictor: η = X β
    - Mean response: μ = sigmoid(η)
    - Gradient: g = Xᵀ (y − μ)
    - Hessian: H = Xᵀ W X, with diagonal W = μ ⊙ (1 − μ)
    - Update: β ← β + H⁻¹ g
    Stopping criterion: max(|g|) <= TOL or MAX_ITER reached.

    Returns coefficients, Hessian inverse, log-likelihood, n_obs, and y_sum.
    """
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

    total_n_obs_arr = agg_client.sum(np.array([float(n_obs_local)], dtype=float))
    total_y_sum_arr = agg_client.sum(np.array([float(y_sum_local)], dtype=float))
    total_n_obs = int(total_n_obs_arr[0])
    total_y_sum = float(total_y_sum_arr[0])

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

        grad_arr = agg_client.sum(grad_local)
        H_arr = agg_client.sum(H_local)
        ll_arr = agg_client.sum(np.array([float(ll_local)], dtype=float))

        grad = np.asarray(grad_arr, dtype=float)
        H = np.asarray(H_arr, dtype=float)
        ll = float(np.asarray(ll_arr, dtype=float).reshape(-1)[0])

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


def compute_logistic_summary(
    *,
    coefficients: np.ndarray,
    h_inv: np.ndarray,
    ll: float,
    n_obs: int,
    y_sum: float,
    alpha: float,
) -> Dict:
    """
    Summaries matching the legacy logistic regression output.

    Formulas:
    - SE(β̂) = sqrt(diag(H⁻¹))
    - z = β̂ / SE(β̂); two-sided p = 2 * (1 - Φ(|z|))
    - CI = β̂ ± z_{1-α/2} * SE
    - df_model = p - 1 (excluding intercept); df_resid = n - p
    - Null LL: ll0 = y_sum * log(ȳ) + (n - y_sum) * log(1 - ȳ)
    - AIC = 2p - 2 ll ; BIC = log(n) * p - 2 ll
    - Pseudo R²: McFadden = 1 - ll/ll0 ; Cox–Snell = 1 - exp(2 (ll0 - ll)/n)
    """
    stderr = np.sqrt(np.diag(h_inv))
    z_scores = np.divide(
        coefficients, stderr, out=np.zeros_like(coefficients), where=stderr != 0
    )
    pvalues = stats.norm.sf(np.abs(z_scores)) * 2

    z_crit = stats.norm.ppf(1 - alpha / 2)
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

    return {
        "n_obs": int(n_obs),
        "coefficients": coefficients.tolist(),
        "stderr": stderr.tolist(),
        "lower_ci": lower_ci,
        "upper_ci": upper_ci,
        "z_scores": z_scores.tolist(),
        "pvalues": pvalues.tolist(),
        "df_model": int(df_model),
        "df_resid": int(df_resid),
        "r_squared_cs": r2_cs,
        "r_squared_mcf": r2_mcf,
        "ll0": ll0,
        "ll": float(ll),
        "aic": aic,
        "bic": bic,
    }

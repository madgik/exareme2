from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import stats

ALPHA = 0.05


def run_distributed_linear_regression(agg_client, X: np.ndarray, y: np.ndarray) -> Dict:
    """
    Ordinary Least Squares with aggregated sufficient statistics.

    Computations:
    - XᵀX and Xᵀy are summed across workers.
    - β̂ = (XᵀX)⁺ Xᵀy (Moore–Penrose if singular).
    - RSS = Σ (y - Xβ̂)², TSS = Σ (y - ȳ)².
    """
    n_features = X.shape[1]

    xTx_local = X.T @ X if n_features > 0 else np.zeros((0, 0), dtype=float)
    xTy_local = X.T @ y if n_features > 0 else np.zeros((0, 1), dtype=float)
    n_obs_local = float(y.size)
    sum_y_local = float(y.sum())
    sum_sq_y_local = float((y**2).sum())

    if n_features > 0:
        xTx = np.asarray(agg_client.sum(xTx_local), dtype=float)
        xTy = np.asarray(agg_client.sum(xTy_local), dtype=float)
    else:
        xTx = np.zeros((0, 0), dtype=float)
        xTy = np.zeros((0, 1), dtype=float)

    n_obs = int(
        np.asarray(
            agg_client.sum(np.array([n_obs_local], dtype=float)), dtype=float
        ).reshape(-1)[0]
    )
    sum_y = float(
        np.asarray(
            agg_client.sum(np.array([sum_y_local], dtype=float)), dtype=float
        ).reshape(-1)[0]
    )
    sum_sq_y = float(
        np.asarray(
            agg_client.sum(np.array([sum_sq_y_local], dtype=float)), dtype=float
        ).reshape(-1)[0]
    )

    if n_features > 0:
        xTx_inv = np.linalg.pinv(xTx)
        coefficients = xTx_inv @ xTy
        rank = int(np.linalg.matrix_rank(xTx))
    else:
        xTx_inv = np.zeros((0, 0), dtype=float)
        coefficients = np.zeros((0, 1), dtype=float)
        rank = 0

    if n_features > 0 and n_obs_local > 0:
        y_pred_local = X @ coefficients
        resid_local = y - y_pred_local
        rss_local = float(np.dot(resid_local.reshape(-1), resid_local.reshape(-1)))
        sum_abs_resid_local = float(np.abs(resid_local).sum())
    else:
        rss_local = 0.0
        sum_abs_resid_local = 0.0

    rss_arr = agg_client.sum(np.array([rss_local], dtype=float))
    sum_abs_resid_arr = agg_client.sum(np.array([sum_abs_resid_local], dtype=float))
    rss = float(np.asarray(rss_arr, dtype=float).reshape(-1)[0])
    sum_abs_resid = float(np.asarray(sum_abs_resid_arr, dtype=float).reshape(-1)[0])

    if n_obs > 0:
        y_mean = sum_y / n_obs
        tss = sum_sq_y - 2.0 * y_mean * sum_y + n_obs * (y_mean**2)
    else:
        tss = 0.0

    return {
        "coefficients": coefficients.reshape(-1).tolist(),
        "xTx_inv": xTx_inv.tolist(),
        "rss": rss,
        "tss": tss,
        "sum_abs_resid": sum_abs_resid,
        "n_obs": n_obs,
        "rank": rank,
    }


def compute_summary_from_stats(
    *,
    coefficients: np.ndarray,
    xTx_inv: np.ndarray,
    rss: float,
    tss: float,
    sum_abs_resid: float,
    n_obs: int,
    p: int,
) -> Dict:
    """
    Reproduce LinearRegression summary metrics using aggregated stats.

    Formulas:
    - df_resid = n - p - 1, df_model = p
    - RSE = sqrt(RSS / df_resid)
    - Var(β̂) = RSE² * diag((XᵀX)⁻¹); SE = sqrt(diag)
    - t = β̂ / SE; two-sided p = 2 * (1 - F_t(|t|, df_resid))
    - CI = β̂ ± t_{1-α/2, df_resid} * SE
    - R² = 1 - RSS/TSS; adj R² = 1 - (1 - R²) * (n - 1)/df_resid
    - F = ((TSS - RSS)/p) / (RSS/df_resid); p-value from F(df=p, df_resid)
    """
    df_resid = n_obs - p - 1
    df_model = p

    if df_resid <= 0 or n_obs <= 0:
        return {
            "n_obs": int(n_obs),
            "df_resid": float(df_resid),
            "df_model": float(df_model),
            "rse": float("nan"),
            "r_squared": 0.0,
            "r_squared_adjusted": 0.0,
            "f_stat": float("nan"),
            "f_pvalue": float("nan"),
            "coefficients": coefficients.reshape(-1).tolist(),
            "std_err": [float("nan")] * len(coefficients),
            "t_stats": [float("nan")] * len(coefficients),
            "pvalues": [float("nan")] * len(coefficients),
            "lower_ci": [float("nan")] * len(coefficients),
            "upper_ci": [float("nan")] * len(coefficients),
        }

    rse = float(np.sqrt(rss / df_resid))

    diag = np.diag(xTx_inv) if xTx_inv.size else np.array([], dtype=float)
    std_err = np.sqrt(np.maximum(diag, 0.0) * (rse**2))

    coeff_flat = coefficients.reshape(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = np.divide(
            coeff_flat,
            std_err,
            out=np.zeros_like(coeff_flat, dtype=float),
            where=std_err != 0,
        )

    t_p_values = stats.t.sf(np.abs(t_stats), df=df_resid) * 2.0

    t_crit = stats.t.ppf(1.0 - ALPHA / 2.0, df=df_resid)
    lower_ci = (coeff_flat - t_crit * std_err).tolist()
    upper_ci = (coeff_flat + t_crit * std_err).tolist()

    if tss > 0:
        r_squared = 1.0 - (rss / tss)
    else:
        r_squared = 0.0
    r_squared_adjusted = 1.0 - (1.0 - r_squared) * (n_obs - 1) / df_resid

    if rss == 0.0 or p == 0:
        f_stat = float("inf")
        f_pvalue = 0.0
    else:
        f_stat = float((tss - rss) * df_resid / (p * rss))
        f_pvalue = float(stats.f.sf(f_stat, dfn=p, dfd=df_resid))

    return {
        "n_obs": int(n_obs),
        "df_resid": float(df_resid),
        "df_model": float(df_model),
        "rse": rse,
        "r_squared": float(r_squared),
        "r_squared_adjusted": float(r_squared_adjusted),
        "f_stat": f_stat,
        "f_pvalue": f_pvalue,
        "coefficients": coeff_flat.tolist(),
        "std_err": std_err.tolist(),
        "t_stats": t_stats.tolist(),
        "pvalues": t_p_values.tolist(),
        "lower_ci": lower_ci,
        "upper_ci": upper_ci,
    }

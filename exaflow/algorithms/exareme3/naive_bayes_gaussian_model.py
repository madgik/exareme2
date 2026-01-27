from typing import List

import numpy as np

from exaflow.algorithms.exareme3.lazy_aggregation import lazy_agg

VAR_SMOOTHING = 1e-9


class GaussianNB:
    """
    Minimal estimator encapsulating the Gaussian Naive Bayes aggregation logic
    so algorithms and tests can reuse the same secure-aggregation routines.
    """

    def __init__(
        self,
        y_var: str,
        x_vars: List[str],
        labels: List[str],
        var_smoothing: float = VAR_SMOOTHING,
    ):
        self.y_var = y_var
        self.x_vars = list(x_vars)
        self.labels = list(labels)
        self.var_smoothing = float(var_smoothing)
        self.theta = None
        self.var = None
        self.class_count = None
        self.class_prior = None
        self.total_n_obs = 0.0

    @lazy_agg
    def fit(self, df, agg_client):
        import pandas as pd

        n_classes = len(self.labels)
        n_features = len(self.x_vars)

        counts_local = np.zeros((n_classes, n_features), dtype=float)
        sums_local = np.zeros_like(counts_local)
        sums_sq_local = np.zeros_like(counts_local)

        if df.shape[0] > 0 and self.y_var in df.columns:
            data = df[self.x_vars].copy()
            data[self.y_var] = pd.Categorical(df[self.y_var], categories=self.labels)

            def sum_sq(x):
                return (x**2).sum()

            agg = (
                data.groupby(by=self.y_var, observed=False)
                .agg(["count", "sum", sum_sq])
                .swaplevel(axis=1)
            )

            counts = agg.xs("count", axis=1).reindex(self.labels).fillna(0.0)
            sums = agg.xs("sum", axis=1).reindex(self.labels).fillna(0.0)
            sums_sq = agg.xs("sum_sq", axis=1).reindex(self.labels).fillna(0.0)

            counts_local = counts.to_numpy(dtype=float)
            sums_local = sums.to_numpy(dtype=float)
            sums_sq_local = sums_sq.to_numpy(dtype=float)

        counts_global = np.asarray(agg_client.sum(counts_local), dtype=float)
        sums_global = np.asarray(agg_client.sum(sums_local), dtype=float)
        sums_sq_global = np.asarray(agg_client.sum(sums_sq_local), dtype=float)

        if counts_global.size == 0:
            self.theta = np.zeros((0, n_features), dtype=float)
            self.var = np.zeros((0, n_features), dtype=float)
            self.class_count = np.zeros(0, dtype=float)
            self.class_prior = np.zeros(0, dtype=float)
            self.total_n_obs = 0.0
            self.labels = []
            return self

        class_count_full = counts_global[:, 0]
        self.total_n_obs = float(class_count_full.sum())

        keep_mask = class_count_full > 0
        if not np.any(keep_mask):
            self.theta = np.zeros((0, n_features), dtype=float)
            self.var = np.zeros((0, n_features), dtype=float)
            self.class_count = np.zeros(0, dtype=float)
            self.class_prior = np.zeros(0, dtype=float)
            self.labels = []
            return self

        counts_eff = counts_global[keep_mask, :]
        sums_eff = sums_global[keep_mask, :]
        sums_sq_eff = sums_sq_global[keep_mask, :]

        means = sums_eff / counts_eff
        var = (
            sums_sq_eff - 2 * means * sums_eff + counts_eff * (means**2)
        ) / counts_eff

        var_max = var.max() if var.size else 0.0
        epsilon = self.var_smoothing * var_max
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            epsilon = self.var_smoothing
        var = np.clip(var, epsilon, None)

        class_count_eff = class_count_full[keep_mask]
        class_sum = class_count_eff.sum()
        if class_sum == 0.0:
            prior = np.ones_like(class_count_eff, dtype=float) / len(class_count_eff)
        else:
            prior = class_count_eff / class_sum

        labels_arr = np.asarray(self.labels, dtype=object)
        self.labels = labels_arr[keep_mask].tolist()
        self.theta = means.astype(float, copy=False)
        self.var = var.astype(float, copy=False)
        self.class_count = class_count_eff.astype(float, copy=False)
        self.class_prior = prior.astype(float, copy=False)
        return self

    def predict_proba(self, X_df):
        from scipy import stats as scipy_stats

        if self.theta is None or self.var is None or self.class_prior is None:
            raise ValueError("GaussianNB is not fitted yet.")

        if X_df.shape[0] == 0 or len(self.labels) == 0:
            return np.zeros((0, len(self.labels)), dtype=float)

        X = X_df[self.x_vars].to_numpy(dtype=float)
        theta = self.theta[np.newaxis, :, :]
        sigma = np.sqrt(self.var)[np.newaxis, :, :]

        factors = scipy_stats.norm.pdf(X[:, np.newaxis, :], loc=theta, scale=sigma)
        likelihood = factors.prod(axis=2)

        prior = self.class_prior[np.newaxis, :]
        unnormalized_post = prior * likelihood
        denom = unnormalized_post.sum(axis=1, keepdims=True)
        denom[denom == 0.0] = 1.0
        posterior = unnormalized_post / denom
        return posterior

    def predict(self, X_df):
        posterior = self.predict_proba(X_df)
        if posterior.shape[0] == 0:
            return np.asarray([], dtype=object)
        idx = posterior.argmax(axis=1)
        labels_arr = np.asarray(self.labels, dtype=object)
        return labels_arr[idx]

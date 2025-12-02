from typing import Dict
from typing import List

import numpy as np
from sklearn.preprocessing import OrdinalEncoder

ALPHA = 1.0


class CategoricalNB:
    """
    Minimal estimator-style class encapsulating the categorical Naive Bayes
    secure-aggregation logic so multiple algorithms/tests can reuse it without
    re-registering UDFs.
    """

    def __init__(self, y_var: str, x_vars: List[str], categories: Dict[str, List[str]]):
        self.y_var = y_var
        self.x_vars = list(x_vars)
        self.categories = categories
        self.labels = list(categories[y_var])
        self.class_count = None
        self.category_count = None

    def fit(self, df, agg_client):
        import pandas as pd

        y_var = self.y_var
        class_cats = self.labels
        n_classes = len(class_cats)

        if df.shape[0] == 0 or y_var not in df.columns:
            class_count_local = np.zeros(n_classes, dtype=float)
        else:
            class_count_series = (
                df.groupby(y_var, observed=False)
                .size()
                .reindex(class_cats, fill_value=0)
            )
            class_count_local = class_count_series.to_numpy(dtype=float)

        class_count_full = np.asarray(agg_client.sum(class_count_local), dtype=float)

        category_count_full = {}
        for xvar in self.x_vars:
            feat_cats = self.categories[xvar]
            if df.shape[0] == 0 or xvar not in df.columns:
                counts_matrix_local = np.zeros((n_classes, len(feat_cats)), dtype=float)
            else:
                idx = pd.MultiIndex.from_product(
                    [class_cats, feat_cats], names=[y_var, xvar]
                )
                counts_series = (
                    df.groupby([y_var, xvar], observed=False)
                    .size()
                    .reindex(idx, fill_value=0)
                )
                counts_matrix_local = counts_series.to_numpy(dtype=float).reshape(
                    (n_classes, len(feat_cats))
                )
            aggregated_counts = agg_client.sum(counts_matrix_local)
            category_count_full[xvar] = np.asarray(aggregated_counts, dtype=float)

        keep_mask = class_count_full > 0
        labels_arr = np.asarray(self.labels, dtype=object)
        self.class_count = class_count_full[keep_mask]
        self.labels = labels_arr[keep_mask].tolist()
        self.category_count = {
            xvar: counts[keep_mask, :] for xvar, counts in category_count_full.items()
        }
        return self

    def predict_proba(self, X_df):
        if self.class_count is None or self.category_count is None:
            raise ValueError("CategoricalNB is not fitted yet.")

        if X_df.shape[0] == 0:
            return np.zeros((0, len(self.labels)), dtype=float)

        feat_categories_ordered = [self.categories[xv] for xv in self.x_vars]
        encoder = OrdinalEncoder(categories=feat_categories_ordered, dtype=int)
        X_enc = encoder.fit_transform(X_df[self.x_vars])

        category_count_list = [self.category_count[xv] for xv in self.x_vars]
        n_feat = np.stack([cc[:, xi] for cc, xi in zip(category_count_list, X_enc.T)])

        n_class = self.class_count[np.newaxis, :, np.newaxis]
        n_cat = np.array([len(cats) for cats in feat_categories_ordered], dtype=float)[
            :, np.newaxis, np.newaxis
        ]

        factors = (n_feat + ALPHA) / (n_class + ALPHA * n_cat)
        likelihood = factors.prod(axis=0).T

        class_sum = self.class_count.sum()
        if class_sum == 0.0:
            prior = np.ones_like(self.class_count, dtype=float) / len(self.class_count)
        else:
            prior = self.class_count / class_sum

        unnormalized_post = prior * likelihood
        denom = unnormalized_post.sum(axis=1, keepdims=True)
        denom[denom == 0.0] = 1.0
        posterior = unnormalized_post / denom
        return posterior

    def predict(self, X_df):
        posterior = self.predict_proba(X_df)
        labels = np.asarray(self.labels)
        if posterior.shape[0] == 0:
            return np.asarray([], dtype=labels.dtype)
        idx = posterior.argmax(axis=1)
        return labels[idx]

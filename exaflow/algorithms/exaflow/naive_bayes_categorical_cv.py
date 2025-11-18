from typing import Dict
from typing import List

import numpy as np
from pydantic import BaseModel

from exaflow.algorithms.exaflow.algorithm import Algorithm
from exaflow.algorithms.exaflow.exaflow_registry import exaflow_udf
from exaflow.algorithms.exaflow.naive_bayes_common import make_naive_bayes_result
from exaflow.algorithms.exaflow.naive_bayes_common import (
    multiclass_classification_metrics,
)
from exaflow.algorithms.exaflow.naive_bayes_common import (
    multiclass_classification_summary,
)
from exaflow.worker_communication import BadUserInput

ALGORITHM_NAME = "naive_bayes_categorical_cv"
ALPHA = 1.0


class CategoricalNBResult(BaseModel):
    # This mirrors what make_naive_bayes_result returns, but we don’t rely
    # on it directly here; Algorithm.run returns whatever
    # make_naive_bayes_result creates.
    # Kept only for clarity / typing; not strictly required by exaflow.
    confusion_matrix: dict
    labels: List[str]
    summary: dict


class CategoricalNBAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self, metadata: dict):
        """
        Cross-validated categorical Naive Bayes using exaflow + aggregation server.

        Mirrors the original exaflow CategoricalNBAlgorithm + CategoricalNB
        logic, but implemented as a single exaflow UDF that:
          * does K-fold CV locally on each worker,
          * uses agg_client to aggregate training counts and confusion matrices,
          * returns per-fold global confusion matrices & n_obs.
        Then we reconstruct the summary using the same metrics helpers and
        make_naive_bayes_result.
        """
        if not self.inputdata.y:
            raise BadUserInput("Naive Bayes CV requires a dependent variable.")
        if not self.inputdata.x:
            raise BadUserInput("Naive Bayes CV requires at least one covariate.")

        y_var = self.inputdata.y[0]
        x_vars = list(self.inputdata.x)

        # Require all variables to be categorical
        non_cat = [v for v in [*x_vars, y_var] if not metadata[v]["is_categorical"]]
        if non_cat:
            raise BadUserInput(
                "Naive Bayes categorical CV only supports categorical variables. "
                f"Non-categorical variables: {', '.join(non_cat)}"
            )

        n_splits = self.parameters.get("n_splits")
        if not isinstance(n_splits, int) or n_splits <= 1:
            raise BadUserInput(
                "Parameter 'n_splits' must be an integer greater than 1."
            )

        # Build sorted category lists to match sklearn / original implementation
        all_vars = x_vars + [y_var]
        categories: Dict[str, List[str]] = {
            var: list(sorted(metadata[var]["enumerations"].keys())) for var in all_vars
        }
        labels = categories[y_var]

        # 1) Per-worker check: n_obs >= n_splits
        check_results = self.engine.run_algorithm_udf(
            func=naive_bayes_categorical_cv_check_local,
            positional_args={
                "inputdata": self.inputdata.json(),
                "y_var": y_var,
                "n_splits": int(n_splits),
            },
        )
        if not all(res["ok"] for res in check_results):
            raise BadUserInput(
                "Cross validation cannot run because some of the workers "
                "participating in the experiment have a number of observations "
                f"smaller than the number of splits, {n_splits}."
            )

        # 2) Run distributed CV with aggregation server
        udf_results = self.engine.run_algorithm_udf(
            func=naive_bayes_categorical_cv_local_step,
            positional_args={
                "inputdata": self.inputdata.json(),
                "y_var": y_var,
                "x_vars": x_vars,
                "categories": categories,
                "n_splits": int(n_splits),
            },
        )

        metrics = udf_results[0]  # identical on all workers

        labels = metrics["labels"]
        confmats = [np.asarray(cm, dtype=float) for cm in metrics["confmats"]]
        n_obs = [int(v) for v in metrics["n_obs"]]

        # Aggregate across folds using the original helpers
        total_confmat = sum(confmats)  # element-wise sum

        per_fold_metrics = [
            multiclass_classification_metrics(confmat) for confmat in confmats
        ]
        summary = multiclass_classification_summary(per_fold_metrics, labels, n_obs)

        # Use the same helper as the Gaussian NB CV to build the final result
        result = make_naive_bayes_result(total_confmat, labels, summary)
        return result


# ---------------------------------------------------------------------------
# Helper UDFs
# ---------------------------------------------------------------------------


@exaflow_udf()
def naive_bayes_categorical_cv_check_local(inputdata, y_var, n_splits):
    """
    Check on each worker whether the number of observations is at least n_splits.
    """
    from exaflow.algorithms.exaflow.data_loading import load_algorithm_dataframe

    data = load_algorithm_dataframe(inputdata, dropna=True)
    if y_var in data.columns:
        n_obs = int(data[y_var].dropna().shape[0])
    else:
        n_obs = 0
    return {"ok": bool(n_obs >= int(n_splits)), "n_obs": n_obs}


@exaflow_udf(with_aggregation_server=True)
def naive_bayes_categorical_cv_local_step(
    inputdata,
    agg_client,
    y_var,
    x_vars,
    categories,
    n_splits,
):
    """
    Exaflow UDF that performs K-fold cross-validation for categorical
    Naive Bayes with secure aggregation.
    """
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import OrdinalEncoder

    from exaflow.algorithms.exaflow.data_loading import load_algorithm_dataframe

    n_splits = int(n_splits)
    data = load_algorithm_dataframe(inputdata, dropna=True)

    # --- NEW: ensure we don't end up with duplicated columns ---
    # Build unique list of columns: x_vars + y_var (in that order)
    cols = list(dict.fromkeys(list(x_vars) + [y_var]))

    # Restrict to relevant columns and drop duplicate column names if any
    data = data[cols].copy()

    # Build categorical columns directly on the same DataFrame
    class_cats = categories[y_var]
    data[y_var] = pd.Categorical(data[y_var], categories=class_cats)
    for xvar in x_vars:
        # Skip any xvar that for some reason is not in data after de-dup
        if xvar not in data.columns:
            continue
        data[xvar] = pd.Categorical(data[xvar], categories=categories[xvar])

    df = data  # fully categorical now

    n_rows = df.shape[0]
    if n_rows == 0:
        return {
            "labels": class_cats,
            "confmats": [],
            "n_obs": [],
        }

    if n_rows < n_splits:
        return {
            "labels": class_cats,
            "confmats": [],
            "n_obs": [],
        }

    # Prepare CV splitter
    idx = np.arange(n_rows)
    kf = KFold(n_splits=n_splits, shuffle=False)

    n_classes = len(class_cats)

    confmats_global: List[np.ndarray] = []
    n_obs_per_fold: List[int] = []

    # Precompute some structures
    class_cats_arr = np.array(class_cats)

    for train_idx, test_idx in kf.split(idx):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        # --------------------------
        # Training: global NB counts
        # --------------------------
        # Class counts
        class_count_series = (
            train_df.groupby(y_var).size().reindex(class_cats, fill_value=0)
        )
        class_count_local = class_count_series.to_numpy(dtype=float)

        class_count_global_list = agg_client.sum(class_count_local.tolist())
        class_count_global = np.asarray(class_count_global_list, dtype=float)
        total_train_n = int(class_count_global.sum())

        # Per-feature category counts (class x category)
        category_count_global_per_var: Dict[str, np.ndarray] = {}
        for xvar in x_vars:
            feat_cats = categories[xvar]
            idx_full = pd.MultiIndex.from_product(
                [class_cats, feat_cats], names=[y_var, xvar]
            )
            counts_series = (
                train_df.groupby([y_var, xvar]).size().reindex(idx_full, fill_value=0)
            )
            counts_matrix_local = counts_series.to_numpy(dtype=float).reshape(
                (n_classes, len(feat_cats))
            )
            flat_local = counts_matrix_local.ravel().tolist()
            flat_global = agg_client.sum(flat_local)
            counts_matrix_global = np.asarray(flat_global, dtype=float).reshape(
                (n_classes, len(feat_cats))
            )
            category_count_global_per_var[xvar] = counts_matrix_global

        if total_train_n == 0:
            # Degenerate; no training data in this fold globally
            confmats_global.append(np.zeros((n_classes, n_classes), dtype=float))
            n_obs_per_fold.append(0)
            continue

        # --------------------------
        # Prediction on test set
        # --------------------------
        if test_df.shape[0] == 0:
            confmats_global.append(np.zeros((n_classes, n_classes), dtype=float))
            n_obs_per_fold.append(total_train_n)
            continue

        # Prepare encoded X for test
        feat_categories_ordered = [categories[xv] for xv in x_vars]
        X_test = test_df[x_vars]
        X_enc = OrdinalEncoder(
            categories=feat_categories_ordered,
            dtype=int,
        ).fit_transform(X_test)

        # category_count list, shaped (n_features, n_classes, n_cats_feature)
        category_count_list = [category_count_global_per_var[xv] for xv in x_vars]

        # Following the original _predict_proba_local logic
        # Build n_feat tensor: (n_features, n_classes, n_samples)
        n_feat = np.stack([cc[:, xi] for cc, xi in zip(category_count_list, X_enc.T)])

        n_class = class_count_global[np.newaxis, :, np.newaxis]
        n_cat = np.array(
            [len(cats) for cats in feat_categories_ordered],
            dtype=float,
        )[:, np.newaxis, np.newaxis]

        factors = (n_feat + ALPHA) / (n_class + ALPHA * n_cat)
        likelihood = factors.prod(axis=0).T  # (n_samples, n_classes)

        prior = class_count_global / class_count_global.sum()
        unnormalized_post = prior * likelihood
        posterior = unnormalized_post / unnormalized_post.sum(axis=1, keepdims=True)

        # --------------------------
        # Confusion matrix (global)
        # --------------------------
        y_true_cat = test_df[y_var]
        # Map to indices 0..C-1, ignore NaNs
        true_codes = y_true_cat.cat.codes.to_numpy()
        valid_mask = true_codes >= 0

        if not np.any(valid_mask):
            confmat_local = np.zeros((n_classes, n_classes), dtype=float)
        else:
            true_idx = true_codes[valid_mask]
            pred_idx = posterior[valid_mask].argmax(axis=1)

            confmat_local = np.zeros((n_classes, n_classes), dtype=float)
            np.add.at(confmat_local, (true_idx, pred_idx), 1.0)

        # Aggregate confusion matrix
        flat_conf_local = confmat_local.ravel().tolist()
        flat_conf_global = agg_client.sum(flat_conf_local)
        confmat_global = np.asarray(flat_conf_global, dtype=float).reshape(
            (n_classes, n_classes)
        )

        confmats_global.append(confmat_global)
        n_obs_per_fold.append(total_train_n)

    return {
        "labels": class_cats_arr.tolist(),
        "confmats": [cm.tolist() for cm in confmats_global],
        "n_obs": n_obs_per_fold,
    }

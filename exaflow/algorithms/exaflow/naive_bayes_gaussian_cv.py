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

ALGORITHM_NAME = "naive_bayes_gaussian_cv"
VAR_SMOOTHING = 1e-9  # same as original GaussianNB _fit_global


class GaussianNBResult(BaseModel):
    # Only here for clarity; Algorithm.run actually returns
    # whatever make_naive_bayes_result returns (NBResult).
    confusion_matrix: dict
    classification_summary: dict


class GaussianNBAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self, metadata: dict):
        """
        Cross-validated Gaussian Naive Bayes using exaflow + aggregation server.

        Mirrors the original exaflow GaussianNBAlgorithm, but bundles the
        cross-validation + fitting + prediction into exaflow UDFs:
          * a local check UDF to ensure n_obs >= n_splits per worker
          * a main CV UDF using agg_client for secure aggregation of
            training statistics and confusion matrices.
        """
        if not self.inputdata.y:
            raise BadUserInput("Naive Bayes Gaussian CV requires a dependent variable.")
        if not self.inputdata.x:
            raise BadUserInput(
                "Naive Bayes Gaussian CV requires at least one covariate."
            )

        y_var = self.inputdata.y[0]
        x_vars = list(self.inputdata.x)

        # Y must be categorical, as in the original GaussianNB
        if not metadata[y_var]["is_categorical"]:
            raise BadUserInput(
                f"Dependent variable {y_var!r} must be categorical for Gaussian NB."
            )

        n_splits = self.parameters.get("n_splits")
        if not isinstance(n_splits, int) or n_splits <= 1:
            raise BadUserInput(
                "Parameter 'n_splits' must be an integer greater than 1."
            )

        # Sorted class labels to match sklearn/original implementation
        label_dict = metadata[y_var]["enumerations"]
        labels = sorted(label_dict.keys())

        # 1) Per-worker feasibility check
        check_results = self.engine.run_algorithm_udf(
            func=gaussian_nb_cv_check_local,
            positional_args={
                "inputdata": self.inputdata.json(),
                "y_var": y_var,
                "n_splits": int(n_splits),
            },
        )

        total_n_obs = sum(res["n_obs"] for res in check_results)

        if total_n_obs < n_splits:
            raise BadUserInput(
                "Cross validation cannot run because the total number of "
                f"observations ({total_n_obs}) is smaller than the number of "
                f"splits, {n_splits}."
            )

        # 2) Run CV UDF (with aggregation server)
        udf_results = self.engine.run_algorithm_udf(
            func=gaussian_nb_cv_local_step,
            positional_args={
                "inputdata": self.inputdata.json(),
                "y_var": y_var,
                "x_vars": x_vars,
                "labels": labels,
                "n_splits": int(n_splits),
            },
        )

        metrics = udf_results[0]  # identical on all workers

        confmats = [np.asarray(cm, dtype=float) for cm in metrics["confmats"]]
        n_obs = [int(v) for v in metrics["n_obs"]]

        # Aggregate confusion matrix across folds
        total_confmat = (
            sum(confmats)
            if confmats
            else np.zeros((len(labels), len(labels)), dtype=float)
        )

        # Compute per-fold metrics and summary, using original helpers
        per_fold_metrics = [
            multiclass_classification_metrics(confmat) for confmat in confmats
        ]
        summary = multiclass_classification_summary(per_fold_metrics, labels, n_obs)

        # Package result using the original helper
        result = make_naive_bayes_result(total_confmat, labels, summary)
        return result


# ---------------------------------------------------------------------------
# Helper UDFs
# ---------------------------------------------------------------------------


@exaflow_udf()
def gaussian_nb_cv_check_local(inputdata, csv_paths, y_var, n_splits):
    """
    Check on each worker whether the number of observations is at least n_splits.
    """
    from exaflow.algorithms.exaflow.data_loading import load_algorithm_dataframe

    data = load_algorithm_dataframe(inputdata, csv_paths, dropna=True)
    if y_var in data.columns:
        n_obs = int(data[y_var].dropna().shape[0])
    else:
        n_obs = 0
    return {"ok": bool(n_obs >= int(n_splits)), "n_obs": n_obs}


@exaflow_udf(with_aggregation_server=True)
def gaussian_nb_cv_local_step(
    inputdata,
    csv_paths,
    agg_client,
    y_var,
    x_vars,
    labels,
    n_splits,
):
    """
    Exaflow UDF that performs K-fold cross-validation for Gaussian Naive Bayes:

      * Each worker fetches its local data.
      * For each fold:
          - Locally computes per-class statistics (count, sum, sum_sq) on train set.
          - Uses agg_client.sum to aggregate these into global stats.
          - From global stats, builds Gaussian NB parameters (means, variances,
            class priors) with VAR_SMOOTHING.
          - On the local test set, computes posterior probabilities and
            local confusion matrix.
          - Confusion matrices are aggregated across workers via agg_client.sum.
      * Returns global confusion matrices and per-fold n_obs (train size).
    """
    import pandas as pd
    from scipy import stats as scipy_stats
    from sklearn.model_selection import KFold

    from exaflow.algorithms.exaflow.data_loading import load_algorithm_dataframe

    data = load_algorithm_dataframe(inputdata, csv_paths, dropna=True)

    n_splits = int(n_splits)
    class_labels = list(labels)
    n_classes_full = len(class_labels)

    # Restrict to X + y
    cols = list(x_vars) + [y_var]
    data = data[cols].copy()

    n_rows = data.shape[0]
    if n_rows == 0:
        # This worker has no data; still needs to participate logically
        return {
            "confmats": [],
            "n_obs": [],
        }

    if n_rows < n_splits:
        # Should be caught by check UDF, but be defensive
        return {
            "confmats": [],
            "n_obs": [],
        }

    # Make y categorical with fixed label order
    data[y_var] = pd.Categorical(data[y_var], categories=class_labels)

    idx = np.arange(n_rows)
    kf = KFold(n_splits=n_splits, shuffle=False)

    confmats_global: List[np.ndarray] = []
    n_obs_per_fold: List[int] = []

    for train_idx, test_idx in kf.split(idx):
        train_df = data.iloc[train_idx]
        test_df = data.iloc[test_idx]

        # --------------------------
        # Training: global stats for NB
        # --------------------------
        if train_df.shape[0] == 0:
            # Degenerate fold
            confmats_global.append(np.zeros((n_classes_full, n_classes_full)))
            n_obs_per_fold.append(0)
            continue

        # Build dataframe like in original _fit_local
        train_data = train_df[x_vars].copy()
        train_data[y_var] = train_df[y_var]

        def sum_sq(x):
            return (x**2).sum()

        agg = train_data.groupby(by=y_var, observed=False).agg(["count", "sum", sum_sq])
        agg = agg.swaplevel(axis=1)

        counts = agg.xs("count", axis=1)
        sums = agg.xs("sum", axis=1)
        sums_sq = agg.xs("sum_sq", axis=1)

        # Reindex to full class label set, fill missing with zeros
        counts = counts.reindex(class_labels).fillna(0.0)
        sums = sums.reindex(class_labels).fillna(0.0)
        sums_sq = sums_sq.reindex(class_labels).fillna(0.0)

        counts_local = counts.to_numpy(dtype=float)
        sums_local = sums.to_numpy(dtype=float)
        sums_sq_local = sums_sq.to_numpy(dtype=float)

        # Aggregate across workers
        counts_glob_flat = agg_client.sum(counts_local.ravel().tolist())
        sums_glob_flat = agg_client.sum(sums_local.ravel().tolist())
        sums_sq_glob_flat = agg_client.sum(sums_sq_local.ravel().tolist())

        counts_global = np.asarray(counts_glob_flat, dtype=float).reshape(
            counts_local.shape
        )
        sums_global = np.asarray(sums_glob_flat, dtype=float).reshape(sums_local.shape)
        sums_sq_global = np.asarray(sums_sq_glob_flat, dtype=float).reshape(
            sums_sq_local.shape
        )

        # class_count is counts for any feature; take first column
        class_count_full = counts_global[:, 0]
        total_train_n = int(class_count_full.sum())

        if total_train_n == 0:
            # No training data globally in this fold
            confmats_global.append(np.zeros((n_classes_full, n_classes_full)))
            n_obs_per_fold.append(0)
            continue

        # Effective classes: those with non-zero training count
        eff_mask = class_count_full > 0
        eff_indices = np.where(eff_mask)[0]
        if eff_indices.size == 0:
            # Should not happen if total_train_n > 0, but be safe
            confmats_global.append(np.zeros((n_classes_full, n_classes_full)))
            n_obs_per_fold.append(0)
            continue

        counts_eff = counts_global[eff_mask, :]
        sums_eff = sums_global[eff_mask, :]
        sums_sq_eff = sums_sq_global[eff_mask, :]

        # Means and variances (per class, per feature) as in _fit_global
        means = sums_eff / counts_eff
        var = (
            sums_sq_eff - 2 * means * sums_eff + counts_eff * (means**2)
        ) / counts_eff

        # Variance smoothing like the original
        epsilon = VAR_SMOOTHING * var.max()
        var = np.clip(var, epsilon, np.inf)

        class_count_eff = class_count_full[eff_mask]
        # Priors for effective classes only
        prior_eff = class_count_eff / class_count_eff.sum()

        # --------------------------
        # Prediction on test set
        # --------------------------
        if test_df.shape[0] == 0:
            confmats_global.append(np.zeros((n_classes_full, n_classes_full)))
            n_obs_per_fold.append(total_train_n)
            continue

        X_test = test_df[x_vars].to_numpy(dtype=float)
        # X_test shape: (n_samples, n_features)
        # theta, var shapes: (n_eff_classes, n_features)
        # We want factors[i, c, j] for sample i, class c, feature j

        theta = means  # (n_eff, n_features)
        sigma = np.sqrt(var)  # (n_eff, n_features)

        # Broadcast:
        # X shape: (n_samples, 1, n_features)
        # theta, sigma: (1, n_eff, n_features)
        X_expanded = X_test[:, np.newaxis, :]
        theta_expanded = theta[np.newaxis, :, :]
        sigma_expanded = sigma[np.newaxis, :, :]

        # Normal pdf per feature
        factors = scipy_stats.norm.pdf(
            X_expanded, loc=theta_expanded, scale=sigma_expanded
        )

        # Product over features -> likelihood per (sample, class)
        likelihood = factors.prod(axis=2)  # (n_samples, n_eff)

        # Posterior ~ prior * likelihood
        unnormalized_post = prior_eff[np.newaxis, :] * likelihood
        denom = unnormalized_post.sum(axis=1, keepdims=True)
        # Guard against all-zero rows (e.g., extreme underflow)
        denom[denom == 0.0] = 1.0
        posterior = unnormalized_post / denom  # (n_samples, n_eff)

        # Predicted effective class index per sample
        pred_eff_idx = posterior.argmax(axis=1)  # (n_samples,)
        # Map to full class index
        pred_full_idx = eff_indices[pred_eff_idx]

        # True class indices according to full label list
        y_true_cat = pd.Categorical(test_df[y_var], categories=class_labels)
        true_codes = np.asarray(y_true_cat.codes, dtype=int)  # already an ndarray
        valid_mask = true_codes >= 0

        confmat_local = np.zeros((n_classes_full, n_classes_full), dtype=float)
        if np.any(valid_mask):
            true_idx = true_codes[valid_mask]
            pred_idx = pred_full_idx[valid_mask]
            np.add.at(confmat_local, (true_idx, pred_idx), 1.0)

        # Aggregate confusion matrix across workers
        flat_conf_local = confmat_local.ravel().tolist()
        flat_conf_glob = agg_client.sum(flat_conf_local)
        confmat_global = np.asarray(flat_conf_glob, dtype=float).reshape(
            (n_classes_full, n_classes_full)
        )

        confmats_global.append(confmat_global)
        n_obs_per_fold.append(total_train_n)

    return {
        "confmats": [cm.tolist() for cm in confmats_global],
        "n_obs": n_obs_per_fold,
    }

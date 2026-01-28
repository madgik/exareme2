from typing import List

import numpy as np

from exaflow.algorithms.exareme3.algorithm import Algorithm
from exaflow.algorithms.exareme3.crossvalidation import min_rows_for_cv
from exaflow.algorithms.exareme3.exareme3_registry import exareme3_udf
from exaflow.algorithms.exareme3.naive_bayes_common import NBResult
from exaflow.algorithms.exareme3.naive_bayes_common import make_naive_bayes_result
from exaflow.algorithms.exareme3.naive_bayes_common import (
    multiclass_classification_metrics,
)
from exaflow.algorithms.exareme3.naive_bayes_common import (
    multiclass_classification_summary,
)
from exaflow.algorithms.exareme3.naive_bayes_gaussian_model import GaussianNB
from exaflow.worker_communication import BadUserInput

ALGORITHM_NAME = "naive_bayes_gaussian_cv"
VAR_SMOOTHING = 1e-9  # same as original GaussianNB _fit_global


class GaussianNBAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self, metadata: dict) -> NBResult:
        y_var = self.inputdata.y[0]
        x_vars = list(self.inputdata.x)
        n_splits = self.parameters.get("n_splits")

        # Sorted class labels to match sklearn/original implementation
        label_dict = metadata[y_var]["enumerations"]
        labels = sorted(label_dict.keys())

        # 1) Per-worker feasibility check
        check_results = self.engine.run_algorithm_udf(
            func=gaussian_nb_cv_check_local,
            positional_args={
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

        return make_naive_bayes_result(total_confmat, labels, summary)


# ---------------------------------------------------------------------------
# Helper UDFs
# ---------------------------------------------------------------------------


@exareme3_udf()
def gaussian_nb_cv_check_local(data, y_var, n_splits):
    """
    Check on each worker whether the number of observations is at least n_splits.
    """

    return min_rows_for_cv(data, y_var, n_splits)


@exareme3_udf(with_aggregation_server=True)
def gaussian_nb_cv_local_step(
    agg_client,
    data,
    y_var,
    x_vars,
    labels,
    n_splits,
):
    """
    Exaflow UDF that performs K-fold cross-validation for Gaussian Naive Bayes.
    """
    import pandas as pd
    from sklearn.model_selection import KFold

    n_splits = int(n_splits)
    class_labels = list(labels)
    n_classes_full = len(class_labels)
    n_features = len(x_vars)

    cols = list(dict.fromkeys(list(x_vars) + [y_var]))
    data = data[cols].copy()

    n_rows = data.shape[0]
    conf_shape = (n_classes_full, n_classes_full)
    if n_rows < n_splits:
        confmats_global: List[np.ndarray] = []
        n_obs_per_fold: List[int] = []
        conf_zero = np.zeros(conf_shape, dtype=float)
        empty_df = data.iloc[0:0].copy()
        empty_df[y_var] = pd.Categorical(empty_df[y_var], categories=class_labels)
        for _ in range(n_splits):
            model = GaussianNB(
                y_var=y_var,
                x_vars=x_vars,
                labels=class_labels,
                var_smoothing=VAR_SMOOTHING,
            )
            model.fit(empty_df, agg_client)
            agg_client.sum(conf_zero.ravel())
            confmats_global.append(conf_zero.copy())
            n_obs_per_fold.append(0)
        return {
            "confmats": [cm.tolist() for cm in confmats_global],
            "n_obs": n_obs_per_fold,
        }

    data[y_var] = pd.Categorical(data[y_var], categories=class_labels)
    idx = np.arange(n_rows)
    kf = KFold(n_splits=n_splits, shuffle=False)

    confmats_global: List[np.ndarray] = []
    n_obs_per_fold: List[int] = []
    label_to_idx = {label: idx for idx, label in enumerate(class_labels)}

    for train_idx, test_idx in kf.split(idx):
        train_df = data.iloc[train_idx]
        test_df = data.iloc[test_idx]

        model = GaussianNB(
            y_var=y_var,
            x_vars=x_vars,
            labels=class_labels,
            var_smoothing=VAR_SMOOTHING,
        )
        model.fit(train_df, agg_client)
        total_train_n = int(model.total_n_obs)

        if total_train_n == 0 or test_df.shape[0] == 0:
            conf_zero = np.zeros(conf_shape, dtype=float)
            flat_conf_glob = agg_client.sum(conf_zero.ravel())
            confmat_global = np.asarray(flat_conf_glob, dtype=float).reshape(conf_shape)
            confmats_global.append(confmat_global)
            n_obs_per_fold.append(total_train_n)
            continue

        posterior = model.predict_proba(test_df[x_vars])
        model_labels = list(model.labels)
        if not model_labels:
            conf_zero = np.zeros(conf_shape, dtype=float)
            flat_conf_glob = agg_client.sum(conf_zero.ravel())
            confmat_global = np.asarray(flat_conf_glob, dtype=float).reshape(conf_shape)
            confmats_global.append(confmat_global)
            n_obs_per_fold.append(total_train_n)
            continue

        local_to_global_idx = np.array(
            [label_to_idx[label] for label in model_labels], dtype=int
        )

        y_true_cat = pd.Categorical(test_df[y_var], categories=class_labels)
        true_codes = np.asarray(y_true_cat.codes, dtype=int)
        valid_mask = true_codes >= 0

        confmat_local = np.zeros(conf_shape, dtype=float)
        if np.any(valid_mask):
            true_idx = true_codes[valid_mask]
            pred_local_idx = posterior[valid_mask].argmax(axis=1)
            pred_idx = local_to_global_idx[pred_local_idx]
            np.add.at(confmat_local, (true_idx, pred_idx), 1.0)

        flat_conf_local = confmat_local.ravel().tolist()
        flat_conf_glob = agg_client.sum(flat_conf_local)
        confmat_global = np.asarray(flat_conf_glob, dtype=float).reshape(conf_shape)

        confmats_global.append(confmat_global)
        n_obs_per_fold.append(total_train_n)

    return {
        "confmats": [cm.tolist() for cm in confmats_global],
        "n_obs": n_obs_per_fold,
    }

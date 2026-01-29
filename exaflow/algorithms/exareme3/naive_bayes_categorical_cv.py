from typing import Dict
from typing import List

import numpy as np
from pydantic import BaseModel

from exaflow.algorithms.exareme3.algorithm import Algorithm
from exaflow.algorithms.exareme3.crossvalidation import min_rows_for_cv
from exaflow.algorithms.exareme3.exareme3_registry import exareme3_udf
from exaflow.algorithms.exareme3.naive_bayes_categorical_model import CategoricalNB
from exaflow.algorithms.exareme3.naive_bayes_common import make_naive_bayes_result
from exaflow.algorithms.exareme3.naive_bayes_common import (
    multiclass_classification_metrics,
)
from exaflow.algorithms.exareme3.naive_bayes_common import (
    multiclass_classification_summary,
)
from exaflow.worker_communication import BadUserInput

ALGORITHM_NAME = "naive_bayes_categorical_cv"


class CategoricalNBResult(BaseModel):
    confusion_matrix: dict
    labels: List[str]
    summary: dict


class CategoricalNBAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self, metadata: dict):
        y_var = self.inputdata.y[0]
        x_vars = list(self.inputdata.x)
        n_splits = self.parameters.get("n_splits")

        # Build sorted category lists to match sklearn / original implementation
        all_vars = x_vars + [y_var]
        categories: Dict[str, List[str]] = {
            var: list(sorted(metadata[var]["enumerations"].keys())) for var in all_vars
        }

        # 1) Per-worker check: n_obs >= n_splits
        check_results = self.run_local_udf(
            func=naive_bayes_categorical_cv_check_local,
            kw_args={
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
        udf_results = self.run_local_udf(
            func=naive_bayes_categorical_cv_local_step,
            kw_args={
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


@exareme3_udf()
def naive_bayes_categorical_cv_check_local(data, y_var, n_splits):
    """
    Check on each worker whether the number of observations is at least n_splits.
    """

    return min_rows_for_cv(data, y_var, n_splits)


@exareme3_udf(with_aggregation_server=True)
def naive_bayes_categorical_cv_local_step(
    agg_client,
    data,
    y_var,
    x_vars,
    categories,
    n_splits,
):
    """
    Exaflow UDF that performs K-fold cross-validation for categorical
    Naive Bayes with secure aggregation.
    """
    import pandas as pd
    from sklearn.model_selection import KFold

    n_splits = int(n_splits)

    # Build unique list of columns: x_vars + y_var (in that order)
    cols = list(dict.fromkeys(list(x_vars) + [y_var]))

    # Restrict to relevant columns and drop duplicate column names if any
    data = data[cols].copy()

    class_cats_full = list(categories[y_var])

    if y_var in data.columns:
        class_count_series = (
            data[y_var].value_counts(dropna=True).reindex(class_cats_full, fill_value=0)
        )
        class_count_local = class_count_series.to_numpy(dtype=float)
    else:
        class_count_local = np.zeros(len(class_cats_full), dtype=float)

    class_count_global = np.asarray(agg_client.sum(class_count_local), dtype=float)
    active_mask = class_count_global > 0
    class_cats = [cat for cat, keep in zip(class_cats_full, active_mask) if keep]
    if not class_cats:
        return {
            "labels": [],
            "confmats": [],
            "n_obs": [],
        }

    data[y_var] = pd.Categorical(data[y_var], categories=class_cats)
    for xvar in x_vars:
        # Skip any xvar that for some reason is not in data after de-dup
        if xvar not in data.columns:
            continue
        data[xvar] = pd.Categorical(data[xvar], categories=categories[xvar])

    df = data  # fully categorical now

    n_rows = df.shape[0]
    if n_rows == 0 or n_rows < n_splits:
        return {
            "labels": class_cats,
            "confmats": [],
            "n_obs": [],
        }

    # Prepare CV splitter
    idx = np.arange(n_rows)
    kf = KFold(n_splits=n_splits, shuffle=False)

    n_classes = len(class_cats)
    label_to_idx = {label: idx for idx, label in enumerate(class_cats)}

    confmats_global: List[np.ndarray] = []
    n_obs_per_fold: List[int] = []

    # Precompute some structures
    class_cats_arr = np.array(class_cats)

    for train_idx, test_idx in kf.split(idx):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        model = CategoricalNB(
            y_var=y_var,
            x_vars=x_vars,
            categories=categories,
        )
        model.fit(train_df, agg_client)
        total_train_n = int(model.class_count.sum())

        if total_train_n == 0:
            conf_zero = np.zeros((n_classes, n_classes), dtype=float)
            flat_conf_global = agg_client.sum(conf_zero.ravel())
            confmat_global = np.asarray(flat_conf_global, dtype=float).reshape(
                (n_classes, n_classes)
            )
            confmats_global.append(confmat_global)
            n_obs_per_fold.append(0)
            continue

        if test_df.shape[0] == 0:
            conf_zero = np.zeros((n_classes, n_classes), dtype=float)
            flat_conf_global = agg_client.sum(conf_zero.ravel())
            confmat_global = np.asarray(flat_conf_global, dtype=float).reshape(
                (n_classes, n_classes)
            )
            confmats_global.append(confmat_global)
            n_obs_per_fold.append(total_train_n)
            continue

        posterior = model.predict_proba(test_df[x_vars])
        model_labels = list(model.labels)
        local_to_global_idx = np.array(
            [label_to_idx[label] for label in model_labels], dtype=int
        )

        y_true_cat = test_df[y_var]
        true_codes = y_true_cat.cat.codes.to_numpy()
        valid_mask = true_codes >= 0

        confmat_local = np.zeros((n_classes, n_classes), dtype=float)
        if np.any(valid_mask):
            true_idx = true_codes[valid_mask]
            pred_local_idx = posterior[valid_mask].argmax(axis=1)
            pred_idx = local_to_global_idx[pred_local_idx]
            np.add.at(confmat_local, (true_idx, pred_idx), 1.0)

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

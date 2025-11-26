import statistics as stats
from typing import List
from typing import NamedTuple
from typing import Optional

import numpy as np
import sklearn.metrics as skm
from pydantic import BaseModel
from scipy.special import expit

from exaflow.aggregation_clients import AggregationType
from exaflow.algorithms.exaflow.algorithm import Algorithm
from exaflow.algorithms.exaflow.exaflow_registry import exaflow_udf
from exaflow.algorithms.exaflow.library.logistic_common import (
    run_distributed_logistic_regression,
)
from exaflow.algorithms.exaflow.metrics import build_design_matrix
from exaflow.algorithms.exaflow.metrics import collect_categorical_levels_from_df
from exaflow.algorithms.exaflow.metrics import construct_design_labels
from exaflow.algorithms.exaflow.metrics import get_dummy_categories
from exaflow.worker_communication import BadUserInput

ALGORITHM_NAME = "logistic_regression_cv"

# ---------------------------------------------------------------------------#
# Models
# ---------------------------------------------------------------------------#


class ConfusionMatrix(BaseModel):
    tp: int
    fp: int
    tn: int
    fn: int

    def __add__(self, other: "ConfusionMatrix") -> "ConfusionMatrix":
        return ConfusionMatrix(
            tp=self.tp + other.tp,
            fp=self.fp + other.fp,
            tn=self.tn + other.tn,
            fn=self.fn + other.fn,
        )

    def ravel(self):
        # Backwards-compatible with original code: tn, fp, fn, tp = confmat.ravel()
        return [self.tn, self.fp, self.fn, self.tp]


class CVClassificationSummary(BaseModel):
    row_names: List[str]
    n_obs: List[Optional[int]]
    accuracy: List[float]
    precision: List[float]
    recall: List[float]
    fscore: List[float]


class ROCCurve(BaseModel):
    name: str
    tpr: List[float]
    fpr: List[float]
    auc: float


class CVLogisticRegressionResult(BaseModel):
    dependent_var: str
    indep_vars: List[str]
    summary: CVClassificationSummary
    confusion_matrix: ConfusionMatrix
    roc_curves: List[ROCCurve]


class BasicMetrics(NamedTuple):
    accuracy: float
    precision: float
    recall: float
    fscore: float


# ---------------------------------------------------------------------------#
# Metric helpers
# ---------------------------------------------------------------------------#


def compute_classification_metrics_from_confmat(
    confmat: ConfusionMatrix,
) -> BasicMetrics:
    tp = confmat.tp
    fp = confmat.fp
    tn = confmat.tn
    fn = confmat.fn

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0

    prec_den = tp + fp
    precision = tp / prec_den if prec_den > 0 else 0.0

    rec_den = tp + fn
    recall = tp / rec_den if rec_den > 0 else 0.0

    if precision + recall > 0:
        fscore = 2.0 * precision * recall / (precision + recall)
    else:
        fscore = 0.0

    return BasicMetrics(
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        fscore=float(fscore),
    )


def make_classification_metrics_summary(
    n_splits: int, n_obs: List[int], metrics: List[BasicMetrics]
) -> CVClassificationSummary:
    row_names = [f"fold_{i}" for i in range(1, n_splits + 1)] + ["average", "stdev"]

    accuracy, precision, recall, fscore = zip(*metrics)

    accuracy = list(accuracy) + [stats.mean(accuracy), stats.stdev(accuracy)]
    precision = list(precision) + [stats.mean(precision), stats.stdev(precision)]
    recall = list(recall) + [stats.mean(recall), stats.stdev(recall)]
    fscore = list(fscore) + [stats.mean(fscore), stats.stdev(fscore)]

    return CVClassificationSummary(
        row_names=row_names,
        n_obs=n_obs + [None, None],  # we don't compute average & stderr for n_obs
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        fscore=fscore,
    )


# ---------------------------------------------------------------------------#
# Main Algorithm
# ---------------------------------------------------------------------------#


class LogisticRegressionCVAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self, metadata: dict):
        """
        Cross-validated logistic regression using exaflow.
        """
        if not self.inputdata.y:
            raise BadUserInput("Logistic regression CV requires a dependent variable.")
        if not self.inputdata.x:
            raise BadUserInput(
                "Logistic regression CV requires at least one covariate."
            )

        positive_class = self.parameters.get("positive_class")
        if positive_class is None:
            raise BadUserInput("Parameter 'positive_class' is required.")

        n_splits = self.parameters.get("n_splits")
        if not isinstance(n_splits, int) or n_splits <= 1:
            raise BadUserInput(
                "Parameter 'n_splits' must be an integer greater than 1."
            )

        y_var = self.inputdata.y[0]

        # Identify categorical vs numerical predictors
        categorical_vars = [
            var for var in self.inputdata.x if metadata[var]["is_categorical"]
        ]
        numerical_vars = [
            var for var in self.inputdata.x if not metadata[var]["is_categorical"]
        ]

        # Discover dummy categories from actual data (shared util)
        dummy_categories = get_dummy_categories(
            engine=self.engine,
            inputdata_json=self.inputdata.json(),
            categorical_vars=categorical_vars,
            collect_udf=logistic_collect_categorical_levels_cv,
        )

        indep_var_names = construct_design_labels(
            categorical_vars=categorical_vars,
            dummy_categories=dummy_categories,
            numerical_vars=numerical_vars,
        )

        # 1) Check per-worker that n_obs >= n_splits
        check_results = self.engine.run_algorithm_udf(
            func=logistic_regression_cv_check_local,
            positional_args={
                "inputdata": self.inputdata.json(),
                "y_var": y_var,
                "positive_class": positive_class,
                "n_splits": n_splits,
            },
        )
        if not all(res["ok"] for res in check_results):
            raise BadUserInput(
                "Cross validation cannot run because some of the workers "
                "participating in the experiment have a number of observations "
                f"smaller than the number of splits, {n_splits}."
            )

        # 2) Run distributed logistic CV with aggregation server
        udf_results = self.engine.run_algorithm_udf(
            func=logistic_regression_cv_local_step,
            positional_args={
                "inputdata": self.inputdata.json(),
                "y_var": y_var,
                "positive_class": positive_class,
                "categorical_vars": categorical_vars,
                "numerical_vars": numerical_vars,
                "dummy_categories": dummy_categories,
                "n_splits": n_splits,
            },
        )

        metrics = udf_results[0]

        n_obs_train = [int(v) for v in metrics["n_obs"]]
        tp_list = [int(v) for v in metrics["tp"]]
        fp_list = [int(v) for v in metrics["fp"]]
        tn_list = [int(v) for v in metrics["tn"]]
        fn_list = [int(v) for v in metrics["fn"]]
        roc_tpr = metrics["roc_tpr"]  # list of list
        roc_fpr = metrics["roc_fpr"]  # list of list

        # Per-fold confusion matrices
        fold_confmats = [
            ConfusionMatrix(tp=tp, fp=fp, tn=tn, fn=fn)
            for tp, fp, tn, fn in zip(tp_list, fp_list, tn_list, fn_list)
        ]

        # Total confusion matrix over all folds
        total_confmat = ConfusionMatrix(tp=0, fp=0, tn=0, fn=0)
        for cm in fold_confmats:
            total_confmat += cm

        # Classification metrics per fold
        fold_metrics = [
            compute_classification_metrics_from_confmat(cm) for cm in fold_confmats
        ]
        summary = make_classification_metrics_summary(
            n_splits=n_splits, n_obs=n_obs_train, metrics=fold_metrics
        )

        # ROC curves per fold + AUC
        roc_curves_result: List[ROCCurve] = []
        for i, (tpr, fpr) in enumerate(zip(roc_tpr, roc_fpr)):
            auc_val = float(skm.auc(x=fpr, y=tpr)) if len(tpr) > 1 else 0.0
            roc_curves_result.append(
                ROCCurve(
                    name=f"fold_{i+1}",
                    tpr=tpr,
                    fpr=fpr,
                    auc=auc_val,
                )
            )

        dependent_var = y_var
        return CVLogisticRegressionResult(
            dependent_var=dependent_var,
            indep_vars=indep_var_names,
            summary=summary,
            confusion_matrix=total_confmat,
            roc_curves=roc_curves_result,
        )


# ---------------------------------------------------------------------------#
# Helper UDFs
# ---------------------------------------------------------------------------#


@exaflow_udf()
def logistic_collect_categorical_levels_cv(data, inputdata, categorical_vars):
    """
    Thin UDF wrapper used only to collect categorical levels from workers.
    """

    return collect_categorical_levels_from_df(data, categorical_vars)


@exaflow_udf()
def logistic_regression_cv_check_local(
    data, inputdata, y_var, positive_class, n_splits
):
    """
    Check on each worker whether the number of observations (for y) is at least n_splits.
    """

    n_splits = int(n_splits)

    if y_var in data.columns:
        y = (data[y_var] == positive_class).astype(float)
        n_obs = int(y.dropna().shape[0])
    else:
        n_obs = 0

    return {"ok": bool(n_obs >= n_splits), "n_obs": n_obs}


@exaflow_udf(with_aggregation_server=True)
def logistic_regression_cv_local_step(
    data,
    inputdata,
    agg_client,
    y_var,
    positive_class,
    categorical_vars,
    numerical_vars,
    dummy_categories,
    n_splits,
):
    """
    Run K-fold CV for logistic regression using secure aggregation.

    For each fold:
    - Train a global logistic model via run_distributed_logistic_regression.
    - Compute probabilities on the test set.
    - Aggregate confusion-matrix counts (threshold 0.5).
    - Approximate ROC curve on a fixed grid of thresholds via aggregated counts.
    """
    from sklearn.model_selection import KFold

    n_splits = int(n_splits)

    if data.empty or y_var not in data.columns:
        return {
            "n_obs": [],
            "tp": [],
            "fp": [],
            "tn": [],
            "fn": [],
            "roc_tpr": [],
            "roc_fpr": [],
        }

    # Build design matrix X and binarized y
    X = build_design_matrix(
        data,
        categorical_vars=categorical_vars,
        dummy_categories=dummy_categories,
        numerical_vars=numerical_vars,
    )
    y = (data[y_var] == positive_class).astype(float).to_numpy().reshape(-1, 1)

    n_rows = X.shape[0]
    if n_rows < n_splits:
        # This should have been caught by the check UDF, but be defensive
        return {
            "n_obs": [],
            "tp": [],
            "fp": [],
            "tn": [],
            "fn": [],
            "roc_tpr": [],
            "roc_fpr": [],
        }

    kf = KFold(n_splits=n_splits, shuffle=False)

    n_features = X.shape[1]

    n_obs_train_per_fold = []
    tp_per_fold = []
    fp_per_fold = []
    tn_per_fold = []
    fn_per_fold = []
    roc_tpr_per_fold = []
    roc_fpr_per_fold = []

    # Fixed grid of thresholds for ROC approximation
    thresholds = np.linspace(0.0, 1.0, 101)
    tp_buf = np.empty_like(thresholds)
    fp_buf = np.empty_like(thresholds)
    tn_buf = np.empty_like(thresholds)
    fn_buf = np.empty_like(thresholds)

    for train_idx, test_idx in kf.split(X):
        X_train = X[train_idx, :]
        y_train = y[train_idx, :]
        X_test = X[test_idx, :]
        y_test = y[test_idx, :]

        if X_train.size == 0:
            n_obs_train_per_fold.append(0)
            tp_per_fold.append(0)
            fp_per_fold.append(0)
            tn_per_fold.append(0)
            fn_per_fold.append(0)
            roc_tpr_per_fold.append([0.0] * len(thresholds))
            roc_fpr_per_fold.append([0.0] * len(thresholds))
            continue

        # Train global logistic model on train set
        train_stats = run_distributed_logistic_regression(agg_client, X_train, y_train)
        coeff = np.asarray(train_stats["coefficients"], dtype=float).reshape(
            (n_features, 1)
        )
        n_train = int(train_stats["n_obs"])
        n_obs_train_per_fold.append(n_train)

        if X_test.size == 0:
            tp_per_fold.append(0)
            fp_per_fold.append(0)
            tn_per_fold.append(0)
            fn_per_fold.append(0)
            roc_tpr_per_fold.append([0.0] * len(thresholds))
            roc_fpr_per_fold.append([0.0] * len(thresholds))
            continue

        # Probabilities on test set
        proba_local = expit(X_test @ coeff).reshape(-1)
        y_true_local = y_test.reshape(-1)

        # Confusion matrix at threshold 0.5
        preds05 = (proba_local >= 0.5).astype(int)
        tp_local = int(((preds05 == 1) & (y_true_local == 1)).sum())
        fp_local = int(((preds05 == 1) & (y_true_local == 0)).sum())
        tn_local = int(((preds05 == 0) & (y_true_local == 0)).sum())
        fn_local = int(((preds05 == 0) & (y_true_local == 1)).sum())

        (
            tp_global_arr,
            fp_global_arr,
            tn_global_arr,
            fn_global_arr,
        ) = agg_client.aggregate_batch(
            [
                (AggregationType.SUM, np.array([float(tp_local)], dtype=float)),
                (AggregationType.SUM, np.array([float(fp_local)], dtype=float)),
                (AggregationType.SUM, np.array([float(tn_local)], dtype=float)),
                (AggregationType.SUM, np.array([float(fn_local)], dtype=float)),
            ]
        )
        tp_global = int(tp_global_arr[0])
        fp_global = int(fp_global_arr[0])
        tn_global = int(tn_global_arr[0])
        fn_global = int(fn_global_arr[0])

        tp_per_fold.append(tp_global)
        fp_per_fold.append(fp_global)
        tn_per_fold.append(tn_global)
        fn_per_fold.append(fn_global)

        # ROC curve: approximate via aggregated counts at fixed thresholds
        for i, thr in enumerate(thresholds):
            preds_thr = proba_local >= thr
            tp_buf[i] = float(((preds_thr) & (y_true_local == 1)).sum())
            fp_buf[i] = float(((preds_thr) & (y_true_local == 0)).sum())
            tn_buf[i] = float((~preds_thr & (y_true_local == 0)).sum())
            fn_buf[i] = float((~preds_thr & (y_true_local == 1)).sum())

        (
            tp_list_global,
            fp_list_global,
            tn_list_global,
            fn_list_global,
        ) = agg_client.aggregate_batch(
            [
                (AggregationType.SUM, tp_buf),
                (AggregationType.SUM, fp_buf),
                (AggregationType.SUM, tn_buf),
                (AggregationType.SUM, fn_buf),
            ]
        )

        tp_arr = np.asarray(tp_list_global, dtype=float)
        fp_arr = np.asarray(fp_list_global, dtype=float)
        tn_arr = np.asarray(tn_list_global, dtype=float)
        fn_arr = np.asarray(fn_list_global, dtype=float)

        with np.errstate(divide="ignore", invalid="ignore"):
            tpr = np.divide(
                tp_arr,
                tp_arr + fn_arr,
                out=np.zeros_like(tp_arr),
                where=(tp_arr + fn_arr) > 0,
            )
            fpr = np.divide(
                fp_arr,
                fp_arr + tn_arr,
                out=np.zeros_like(fp_arr),
                where=(fp_arr + tn_arr) > 0,
            )

        roc_tpr_per_fold.append(tpr.tolist())
        roc_fpr_per_fold.append(fpr.tolist())

    return {
        "n_obs": n_obs_train_per_fold,
        "tp": tp_per_fold,
        "fp": fp_per_fold,
        "tn": tn_per_fold,
        "fn": fn_per_fold,
        "roc_tpr": roc_tpr_per_fold,
        "roc_fpr": roc_fpr_per_fold,
    }

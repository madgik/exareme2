import warnings

import numpy
import numpy as np
import pandas as pd

from exareme2.algorithms.in_database.helpers import get_transfer_data
from exareme2.algorithms.in_database.helpers import sum_secure_transfers
from exareme2.algorithms.in_database.udfgen import literal
from exareme2.algorithms.in_database.udfgen import relation
from exareme2.algorithms.in_database.udfgen import secure_transfer
from exareme2.algorithms.in_database.udfgen import udf


def confusion_matrix_binary(engine, ytrue, proba):
    """
    Compute confusion matrix for binary classification

    Parameters
    ----------
    engine : AlgorithmExecutionEngine
        Algorithm execution engine passed in algorithm's `run` function.
    ytrue : relation
        Ground truth (correct) target values.
    proba : relation
        Estimated target probabilities returned by the classifier.

    Returns
    -------
    numpy.array of shape (2, 2):
        Confusion matrix arranged as [[TN, FP], [FN, TP]]
    """
    loctransf = engine.run_udf_on_local_nodes(
        func=_confusion_matrix_binary_local,
        keyword_args={"ytrue": ytrue, "proba": proba},
        share_to_global=[True],
    )
    result = engine.run_udf_on_global_node(
        func=sum_secure_transfers,
        keyword_args={"loctransf": loctransf},
    )
    confmat = get_transfer_data(result)["confmat"]
    return numpy.array(confmat)


@udf(
    ytrue=relation(),
    proba=relation(),
    return_type=secure_transfer(sum_op=True),
)
def _confusion_matrix_binary_local(ytrue, proba):
    ytrue, proba = ytrue.align(proba, axis=0, copy=False)
    ytrue, proba = ytrue.iloc[:, 0].to_numpy(), proba.iloc[:, 0].to_numpy()

    tp = ((proba > 0.5) & (ytrue == 1)).sum()
    tn = ((proba <= 0.5) & (ytrue == 0)).sum()
    fp = ((proba > 0.5) & (ytrue == 0)).sum()
    fn = ((proba <= 0.5) & (ytrue == 1)).sum()

    confmat = [[int(tn), int(fp)], [int(fn), int(tp)]]

    result = {"confmat": {"data": confmat, "type": "int", "operation": "sum"}}
    return result


def roc_curve(engine, ytrue, proba):
    """
    Compute Receiver operating characteristic (ROC) for binary classification

    Parameters
    ----------
    engine : AlgorithmExecutionEngine
        Algorithm execution engine passed in algorithm's `run` function.
    ytrue : relation
        Ground truth (correct) target values.
    proba : relation
        Estimated target probabilities returned by the classifier.

    Returns
    -------
    Tuple[List[float], List[float]]
        A pair of lists of floats, the first being the true positive rate TPR,
        and the second the false positive rate FPR.
    """
    thresholds = numpy.linspace(1.0, 0.0, num=200).tolist()
    loctransf = engine.run_udf_on_local_nodes(
        func=_roc_curve_local,
        keyword_args={"ytrue": ytrue, "proba": proba, "thresholds": thresholds},
        share_to_global=[True],
    )
    global_transfer = engine.run_udf_on_global_node(
        func=sum_secure_transfers,
        keyword_args={"loctransf": loctransf},
    )
    total_counts = get_transfer_data(global_transfer)
    return _get_tpr_fpr_from_counts(total_counts)


@udf(
    ytrue=relation(),
    proba=relation(),
    thresholds=literal(),
    return_type=secure_transfer(sum_op=True),
)
def _roc_curve_local(ytrue, proba, thresholds):
    ytrue, proba = ytrue.align(proba, axis=0, copy=False)
    ytrue, proba = ytrue["ybin"].to_numpy(), proba["proba"].to_numpy()

    tp = numpy.empty(len(thresholds), dtype=int)
    tn = numpy.empty(len(thresholds), dtype=int)
    fp = numpy.empty(len(thresholds), dtype=int)
    fn = numpy.empty(len(thresholds), dtype=int)

    for i, thres in enumerate(thresholds):
        tp[i] = ((proba > thres) & (ytrue == 1)).sum()
        tn[i] = ((proba <= thres) & (ytrue == 0)).sum()
        fp[i] = ((proba > thres) & (ytrue == 0)).sum()
        fn[i] = ((proba <= thres) & (ytrue == 1)).sum()

    result = dict(
        tp={"data": tp.tolist(), "type": "int", "operation": "sum"},
        tn={"data": tn.tolist(), "type": "int", "operation": "sum"},
        fp={"data": fp.tolist(), "type": "int", "operation": "sum"},
        fn={"data": fn.tolist(), "type": "int", "operation": "sum"},
    )
    return result


def _get_tpr_fpr_from_counts(counts):
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]

    tpr = [recall(tpi, fni) for tpi, fni in zip(tp, fn)]
    fpr = [1 - specificity(tni, fpi) for tni, fpi in zip(tn, fp)]
    return tpr, fpr


def compute_classification_metrics(confmat):
    """Compute accuracy, precision, recall and f-score from confusion matrix"""
    tn, fp, fn, tp = confmat.ravel()
    tn, fp, fn, tp = map(int, (tn, fp, fn, tp))  # cast from numpy.int64
    return (
        accuracy(tp, tn, fp, fn),
        precision(tp, fp),
        recall(tp, fn),
        fscore(tp, fp, fn),
    )


def accuracy(tp, tn, fp, fn):
    try:
        return (tp + tn) / (tp + tn + fp + fn)
    except ZeroDivisionError:
        return 0


def precision(tp, fp):
    try:
        return tp / (tp + fp)
    except ZeroDivisionError:
        return 0


def recall(tp, fn):
    try:
        return tp / (tp + fn)
    except ZeroDivisionError:
        return 0


def specificity(tn, fp):
    try:
        return tn / (tn + fp)
    except ZeroDivisionError:
        return 0


def fscore(tp, fp, fn):
    prec = precision(tp, fp)
    rec = recall(tp, fn)
    try:
        return 2 * (prec * rec) / (prec + rec)
    except ZeroDivisionError:
        return 0


def confusion_matrix_multiclass(engine, ytrue, proba, labels):
    """
    Compute confusion matrix for multiclass classification

    Parameters
    ----------
    engine : AlgorithmExecutionEngine
        Algorithm execution engine passed in algorithm's `run` function.
    ytrue : relation
        Ground truth (correct) target values.
    proba : relation
        Estimated target probabilities returned by the classifier.
    labels : list
        List of labels to index the matrix. The list should have length n_labels.

    Returns
    -------
    numpy.array of shape (n_labels, n_labels):
        Confusion matrix whose i-th row and j-th column entry indicates the
        number of samples with true label being i-th class and predicted label
        being j-th class.
    """
    loctransf = engine.run_udf_on_local_nodes(
        func=_confusion_matrix_multiclass_local,
        keyword_args={"ytrue": ytrue, "proba": proba, "labels": labels},
        share_to_global=[True],
    )
    result = engine.run_udf_on_global_node(
        func=sum_secure_transfers,
        keyword_args={"loctransf": loctransf},
    )
    confmat = get_transfer_data(result)["confmat"]
    return numpy.array(confmat)


@udf(
    ytrue=relation(),
    proba=relation(),
    labels=literal(),
    return_type=secure_transfer(sum_op=True),
)
def _confusion_matrix_multiclass_local(ytrue, proba, labels):
    import numpy as np

    ytrue, proba = ytrue.align(proba, axis=0, copy=False)
    ytrue, proba = ytrue.values, proba.values
    labels = np.array(labels)

    ypred = labels[proba.argmax(axis=1)][:, np.newaxis]

    # I need to count the matches and missmatches between predictions and true
    # values independently for each label. In order to do that I create one-hot
    # encondings for the predictions and for the true values. That gives me
    # matrices of shape (n_obs, n_labels). Then, I need to take the cross
    # product between predictions and true values, with respect to the second
    # dimension, in order to have all combinations of predicted label and true
    # label. The trick to take the cross product is to insert a new axis at the
    # right dimension. Finaly, I sum over the first dimension to aggregate all
    # counts.
    pred_onehot = (ypred == labels)[:, np.newaxis, :]  # shape=(n_obs, 1, n_labels)
    true_onehot = (ytrue == labels)[:, :, np.newaxis]  # shape=(n_obs, n_labels, 1)
    confmat = (pred_onehot & true_onehot).sum(axis=0)  # cross prod then sum over n_obs

    result = {"confmat": {"data": confmat.tolist(), "type": "int", "operation": "sum"}}
    return result


def multiclass_classification_metrics(confmat):
    """
    Computes classification metrics from confusion matrix

    The classification metrics are accuracy, precision, recall and fscore.
    These are all computed starting from a multiclass confusion matrix.

    Parameters
    ----------
    confmat : numpy.array of shape (n_labels, n_labels)
        A multiclass confusion matrix

    Returns
    -------
    dict
        A dictionary containing all the classification metrics
    """
    n_labels, _ = confmat.shape

    # In order to compute the classification metrics we first compute the true
    # positives and negatives, and the false positives and negatives. These are
    # computed by summing the relevand subparts of the confusion matrix,
    # explained below case by case.
    # Then, the classification metrics are computed from the TP, TN, FP, FN.

    # True positives are found in the diagonal of the matrix
    tp = np.diag(confmat)

    # True negatives are the sums of the submatrices, complementary to the
    # diagonal elements
    ix_args = [[[i for i in range(n_labels) if i != j]] * 2 for j in range(n_labels)]
    tn = np.array([confmat[np.ix_(*args)].sum() for args in ix_args])

    # For false negatives we sum every row omitting the diagonal elements
    fn_idcs = [
        ([i] * (n_labels - 1), [j for j in range(n_labels) if j != i])
        for i in range(n_labels)
    ]
    fn = np.array([confmat[idx] for idx in fn_idcs]).sum(axis=1)

    # For false positives we sum every column omitting the diagonal elements, hence
    # we need to swap fn indices
    fp_idcs = [(lambda a, b: (b, a))(*idx) for idx in fn_idcs]
    fp = np.array([confmat[idx] for idx in fp_idcs]).sum(axis=1)

    # Divisions by zero raise warnings but we replace NaNs later anyway
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fscore = 2 * (precision * recall) / (precision + recall)

    # Replace NaNs with 0s
    accuracy = np.nan_to_num(accuracy)
    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)
    fscore = np.nan_to_num(fscore)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "fscore": fscore,
    }


def multiclass_classification_summary(metrics, labels, n_obs):
    """Formats the classification metrics into a summary table, represented by
    a nested dict."""
    # Zip values with labels and index by fold
    data = {
        f"fold{i}": {k: dict(zip(labels, v)) for k, v in m.items()}
        for i, m in enumerate(metrics)
    }

    # Reformat nested dict in a format understood by pandas as a multi-index.
    reform = {
        fold_key: {
            (metrics_key, level): val
            for metrics_key, metrics_vals in fold_val.items()
            for level, val in metrics_vals.items()
        }
        for fold_key, fold_val in data.items()
    }
    # Then transpose to convert multi-index dataframe into hierarchical one
    # (multi-index on the columns).
    df = pd.DataFrame(reform).T

    # Append rows for average and stdev of every column
    df.loc["average"] = df.mean()
    df.loc["stdev"] = df.std()

    # Hierarchical dataframe to nested dict
    summary = {level: df.xs(level, axis=1).to_dict() for level in df.columns.levels[0]}

    summary["n_obs"] = {f"fold{i}": int(n) for i, n in enumerate(n_obs)}
    return summary

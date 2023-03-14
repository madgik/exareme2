import numpy

from mipengine.algorithms.helpers import get_transfer_data
from mipengine.algorithms.helpers import sum_secure_transfers
from mipengine.udfgen import literal
from mipengine.udfgen import relation
from mipengine.udfgen import secure_transfer
from mipengine.udfgen import udf


def confusion_matrix(engine, ytrue, proba):
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
        func=_confusion_matrix_local,
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
def _confusion_matrix_local(ytrue, proba):
    ytrue, proba = ytrue.align(proba, axis=0, copy=False)
    ytrue, proba = ytrue["ybin"].to_numpy(), proba["proba"].to_numpy()

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

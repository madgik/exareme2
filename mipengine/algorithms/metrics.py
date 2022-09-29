import numpy

from mipengine.algorithms.helpers import get_transfer_data
from mipengine.algorithms.helpers import sum_secure_transfers
from mipengine.udfgen import literal
from mipengine.udfgen import relation
from mipengine.udfgen import secure_transfer
from mipengine.udfgen import udf


def confusion_matrix(executor, ytrue, proba):
    """
    Compute confusion matrix for binary classification

    Parameters
    ----------
    executor : _AlgorithmExecutionInterface
        Algorithm execution interface passed in algorithm's `run` function.
    ytrue : relation
        Ground truth (correct) target values.
    proba : relation
        Estimated target probabilities returned by the classifier.

    Returns
    -------
    numpy.array of shape (2, 2):
        Confusion matrix arranged as [[TN, FP], [FN, TP]]
    """
    loctransf = executor.run_udf_on_local_nodes(
        func=_confusion_matrix_local,
        keyword_args={"ytrue": ytrue, "proba": proba},
        share_to_global=[True],
    )
    result = executor.run_udf_on_global_node(
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
    import sklearn

    ytrue, proba = ytrue.align(proba, axis=0, copy=False)
    ytruec, probac = ytrue["ybin"], proba["proba"]
    probac = probac.to_numpy()[:, numpy.newaxis]

    ypred = sklearn.preprocessing.binarize(probac, threshold=0.5).reshape(-1)

    # labels are needed below for cases where not all labels appear in ytrue, ypred
    confmat = sklearn.metrics.confusion_matrix(ytruec, ypred, labels=[0, 1])

    result = {"confmat": {"data": confmat.tolist(), "type": "int", "operation": "sum"}}
    return result


def roc_curve(executor, ytrue, proba):
    """
    Compute Receiver operating characteristic (ROC) for binary classification

    Parameters
    ----------
    executor : _AlgorithmExecutionInterface
        Algorithm execution interface passed in algorithm's `run` function.
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
    loctransf = executor.run_udf_on_local_nodes(
        func=_roc_curve_local,
        keyword_args={"ytrue": ytrue, "proba": proba, "thresholds": thresholds},
        share_to_global=[True],
    )
    global_transfer = executor.run_udf_on_global_node(
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
    import sklearn

    ytrue, proba = ytrue.align(proba, axis=0, copy=False)
    ytruec, probac = ytrue["ybin"], proba["proba"]

    probac = probac.to_numpy()[:, numpy.newaxis]
    ytruec = ytruec.to_numpy()

    def binarize_by_threshold(threshold):
        return sklearn.preprocessing.binarize(probac, threshold=threshold).reshape(-1)

    ypreds = numpy.stack([binarize_by_threshold(thres) for thres in thresholds])

    tp = ((ypreds == 1) & (ytruec == 1)).sum(axis=1)
    tn = ((ypreds == 0) & (ytruec == 0)).sum(axis=1)
    fp = ((ypreds == 1) & (ytruec == 0)).sum(axis=1)
    fn = ((ypreds == 0) & (ytruec == 1)).sum(axis=1)

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

import typing as t
import warnings

import numpy as np
import pandas as pd
from pydantic import BaseModel


class ConfusionMatrix(BaseModel):
    """Multiclass confusion matrix model

    Each row of the matrix represents the instances in an actual class while
    each column represents the instances in a predicted class.

    Attributes
    ----------
    data
        Confusion matrix data in row major order
    labels
        Labels of the classes used in classification
    """

    data: t.List[t.List[int]]
    labels: t.List[str]


class MulticlassClassificationSummary(BaseModel):
    """Multiclass classification summary model

    In cross validated multiclass classification, the accuracy, precision,
    recall and fscore are computed for every class, for every fold. The number
    of observations, n_obs, is different for every fold, but doesn't depend on
    the class.

    This produces a hierarchical table. E.g. for two classes cl1, cl2 the table
    has the following form.

    |      | accuracy  | precision |  recall   |  fscore   |       |
    | fold | cl1 | cl2 | cl1 | cl2 | cl1 | cl2 | cl1 | cl2 | n_obs |
    |------+-----------+-----------+-----------+-----------|-------|
    |    1 | ..  | ..  | ..  | ..  | ..  | ..  | ..  | ..  |  ..   |
    |    2 | ..  | ..  | ..  | ..  | ..  | ..  | ..  | ..  |  ..   |

    This table is represented as a collection of mappings. For the hierarchical
    quantities these mappings are nested and have the form
        {"accuracy": {"cl1": ..., "cl2": ...}, ...}
    """

    accuracy: t.Dict[str, t.Dict[str, float]]
    precision: t.Dict[str, t.Dict[str, float]]
    recall: t.Dict[str, t.Dict[str, float]]
    fscore: t.Dict[str, t.Dict[str, float]]
    n_obs: t.Dict[str, int]


class NBResult(BaseModel):
    confusion_matrix: ConfusionMatrix
    classification_summary: MulticlassClassificationSummary


def make_naive_bayes_result(confmat, labels, summary) -> NBResult:
    """Helper to build the NBResult from a confusion matrix + summary dict."""
    confmat_model = ConfusionMatrix(data=confmat.tolist(), labels=labels)
    summary_model = MulticlassClassificationSummary(**summary)
    result = NBResult(
        confusion_matrix=confmat_model,
        classification_summary=summary_model,
    )
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

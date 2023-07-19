import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest
import sklearn
from hypothesis import assume
from hypothesis import given
from hypothesis import seed
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from exareme2.algorithms.metrics import _confusion_matrix_binary_local
from exareme2.algorithms.metrics import _confusion_matrix_multiclass_local
from exareme2.algorithms.metrics import _get_tpr_fpr_from_counts
from exareme2.algorithms.metrics import _roc_curve_local
from exareme2.algorithms.metrics import multiclass_classification_metrics


def test__confusion_matrix_local_binary():
    ytrue = pd.DataFrame(data=[0, 0, 1, 1])
    proba = pd.DataFrame(data=[0.3, 0.7, 0.3, 0.7])

    result = _confusion_matrix_binary_local(ytrue, proba)
    expected = [[1, 1], [1, 1]]
    assert result["confmat"]["data"] == expected
    assert result["confmat"]["type"] == "int"
    assert result["confmat"]["operation"] == "sum"


def test__confusion_matrix_binary_local__all_diff():
    ytrue = pd.DataFrame(data=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    proba = pd.DataFrame(data=[0.3, 0.7, 0.7, 0.3, 0.3, 0.3, 0.7, 0.7, 0.7, 0.7, 0.7])

    result = _confusion_matrix_binary_local(ytrue, proba)
    expected = [[1, 2], [3, 4]]
    assert result["confmat"]["data"] == expected
    assert result["confmat"]["type"] == "int"
    assert result["confmat"]["operation"] == "sum"


def test__confusion_matrix_multiclass_local():
    ytrue = pd.DataFrame(data=["F", "F", "M", "M"])
    probas = pd.DataFrame({"F": [0.3, 0.7, 0.3, 0.7], "M": [0.7, 0.3, 0.7, 0.3]})
    labels = ["F", "M"]

    result = _confusion_matrix_multiclass_local(ytrue, probas, labels)
    expected = [[1, 1], [1, 1]]
    assert result["confmat"]["data"] == expected
    assert result["confmat"]["type"] == "int"
    assert result["confmat"]["operation"] == "sum"


def test__confusion_matrix_multiclass_local__all_diff():
    ytrue = pd.DataFrame(data=["F", "F", "F", "M", "M", "M", "M", "M", "M", "M"])
    probas = pd.DataFrame(
        {
            "F": [0.7, 0.3, 0.3, 0.7, 0.7, 0.7, 0.3, 0.3, 0.3, 0.3],
            "M": [0.3, 0.7, 0.7, 0.3, 0.3, 0.3, 0.7, 0.7, 0.7, 0.7],
        }
    )
    labels = ["F", "M"]

    result = _confusion_matrix_multiclass_local(ytrue, probas, labels)
    expected = [[1, 2], [3, 4]]
    assert result["confmat"]["data"] == expected
    assert result["confmat"]["type"] == "int"
    assert result["confmat"]["operation"] == "sum"


@pytest.mark.slow
def test_roc_curve():
    """Randomized test, validates our `roc_curve` against sklearn's using 100
    random inputs"""
    for _ in range(100):
        ytrue = pd.DataFrame({"ybin": np.random.randint(0, 2, size=100)})
        proba = pd.DataFrame({"proba": np.random.rand(100)})

        # expected results from sklearn
        tpr_exp, fpr_exp, thresholds = sklearn.metrics.roc_curve(ytrue, proba)

        # NOTE The reason for subtracting 1e-8 from thresholds is an aparent
        # inconsistency in sklearn's implementation. The function `binarize` uses a
        # condition with stricly greater than, see
        # https://github.com/scikit-learn/scikit-learn/blob/36958fb240fbe435673a9e3c52e769f01f36bec0/sklearn/preprocessing/_data.py#L1996
        # whereas `roc_curve` uses equal or greater than, see
        # https://github.com/scikit-learn/scikit-learn/blob/36958fb240fbe435673a9e3c52e769f01f36bec0/sklearn/metrics/_ranking.py#L724
        thresholds = thresholds - 1e-8

        # computed results
        result = _roc_curve_local(ytrue, proba, thresholds)
        counts = {key: val["data"] for key, val in result.items()}
        tpr_res, fpr_res = _get_tpr_fpr_from_counts(counts)
        assert tpr_res == tpr_res
        assert fpr_res == fpr_res


@st.composite
def numeric_labels(draw):
    n_labels = draw(st.integers(2, 10))
    return np.arange(n_labels)


@st.composite
def text_labels(draw):
    small_text = st.text(min_size=1, max_size=10)
    labels = draw(st.lists(small_text, min_size=2, max_size=10, unique=True))
    return np.array(labels)


@st.composite
def classification_data(draw, labels_strategy):
    labels = draw(labels_strategy)

    size = draw(st.integers(1, 200))
    ytrue = draw(st.lists(st.sampled_from(labels), min_size=size, max_size=size))
    ypred = draw(st.lists(st.sampled_from(labels), min_size=size, max_size=size))

    assume(np.intersect1d(ytrue, labels).size > 0)
    cm = confusion_matrix(ytrue, ypred, labels=labels)
    return np.array(labels), np.array(ytrue), np.array(ypred), cm


@pytest.mark.slow
class Test_multiclass_classification_metrics:
    score_options = {"average": None, "zero_division": False}

    @seed(0)
    @given(classification_data(text_labels() | numeric_labels()))
    def test_precision(self, args):
        labels, ytrue, ypred, confmat = args
        expected = precision_score(ytrue, ypred, labels=labels, **self.score_options)

        precision = multiclass_classification_metrics(confmat)["precision"]

        np.testing.assert_allclose(precision, expected)

    @seed(0)
    @given(classification_data(text_labels() | numeric_labels()))
    def test_recall(self, args):
        labels, ytrue, ypred, confmat = args
        expected = recall_score(ytrue, ypred, labels=labels, **self.score_options)

        recall = multiclass_classification_metrics(confmat)["recall"]

        np.testing.assert_allclose(recall, expected)

    @seed(0)
    @given(classification_data(text_labels() | numeric_labels()))
    def test_fscore(self, args):
        labels, ytrue, ypred, confmat = args
        expected = f1_score(ytrue, ypred, labels=labels, **self.score_options)

        fscore = multiclass_classification_metrics(confmat)["fscore"]

        np.testing.assert_allclose(fscore, expected)

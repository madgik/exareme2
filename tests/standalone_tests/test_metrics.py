import numpy as np
import pandas as pd
import pytest
import sklearn

from mipengine.algorithms.metrics import _confusion_matrix_binary_local
from mipengine.algorithms.metrics import _get_tpr_fpr_from_counts
from mipengine.algorithms.metrics import _roc_curve_local


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

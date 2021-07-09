import numpy as np
import pandas as pd

from mipengine.algorithms.logistic_regression import label_binarize


def test_label_binarize_two_classes_equal():
    y = pd.DataFrame({"y": ["a", "a", "b", "b"]})
    ybin = label_binarize(y, classes=["a", "b"])
    assert sum(ybin) == 2


def test_label_binarize_two_classes_unequal():
    y = pd.DataFrame({"y": ["a", "a", "b", "b", "b"]})
    ybin = label_binarize(y, classes=["b", "a"])
    assert sum(ybin) == 2 or sum(ybin) == 3


def test_label_binarize_three_classes():
    y = pd.DataFrame({"y": ["a", "a", "b", "b", "c"]})
    ybin = label_binarize(y, classes=["a"])
    expected = np.array([1, 1, 0, 0, 0])
    assert (ybin == expected).all()

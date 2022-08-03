import pandas as pd
import pytest

from mipengine.algorithms.preprocessing import LabelBinarizer


@pytest.fixture
def fake_executor():
    class FakeExecutor:
        run_udf_on_local_nodes = None
        run_udf_on_global_node = None

    return FakeExecutor()


def test_label_binarizer_binary_input(fake_executor):
    y = pd.DataFrame({"y": ["a", "a", "b", "a"]})
    lb = LabelBinarizer(fake_executor, None)

    result = lb._transform_local(y, positive_class="a")
    assert result.ybin.tolist() == [1, 1, 0, 1]

    result = lb._transform_local(y, positive_class="b")
    assert result.ybin.tolist() == [0, 0, 1, 0]


def test_label_binarizer_missing_class(fake_executor):
    y = pd.DataFrame({"y": ["a", "a", "b", "a"]})
    lb = LabelBinarizer(fake_executor, None)

    result = lb._transform_local(y, positive_class="c")
    assert result.ybin.tolist() == [0, 0, 0, 0]


def test_label_binarizer_non_binary_input(fake_executor):
    y = pd.DataFrame({"y": ["a", "b", "c", "a"]})
    lb = LabelBinarizer(fake_executor, None)

    result = lb._transform_local(y, positive_class="a")
    assert result.ybin.tolist() == [1, 0, 0, 1]

    result = lb._transform_local(y, positive_class="b")
    assert result.ybin.tolist() == [0, 1, 0, 0]

    result = lb._transform_local(y, positive_class="c")
    assert result.ybin.tolist() == [0, 0, 1, 0]

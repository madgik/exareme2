from unittest.mock import Mock
from unittest.mock import call
from unittest.mock import sentinel as s

from exareme2.algorithms.exareme2.crossvalidation import cross_validate


class FakeSplitter:
    """Fake splitter used for testing

    Returns the train and test sets for X and y. Each set has size 2 and is
    filled with sentinels, used to verify that the correct methods are called
    with the correct arguments, durring cross-validation."""

    n_splits = 2

    def split(self, *_):
        x_train = [s.x_tr0, s.x_tr1]
        x_test = [s.x_te0, s.x_te1]
        y_train = [s.y_tr0, s.y_tr1]
        y_test = [s.y_te0, s.y_te1]
        return x_train, x_test, y_train, y_test


def test_cross_validate__return_values():
    splitter = FakeSplitter()
    X = y = None
    models = [Mock() for _ in range(splitter.n_splits)]
    mod0_expected_calls = [call.fit(X=s.x_tr0, y=s.y_tr0), call.predict(s.x_te0)]
    mod1_expected_calls = [call.fit(X=s.x_tr1, y=s.y_tr1), call.predict(s.x_te1)]

    y_pred, y_true = cross_validate(X, y, models, splitter, pred_type="values")

    models[0].assert_has_calls(mod0_expected_calls)
    models[1].assert_has_calls(mod1_expected_calls)
    assert len(y_pred) == 2
    assert y_true == [s.y_te0, s.y_te1]


def test_cross_validate__return_probabilities():
    splitter = FakeSplitter()
    X = y = None
    models = [Mock() for _ in range(splitter.n_splits)]
    mod0_expected_calls = [call.fit(X=s.x_tr0, y=s.y_tr0), call.predict_proba(s.x_te0)]
    mod1_expected_calls = [call.fit(X=s.x_tr1, y=s.y_tr1), call.predict_proba(s.x_te1)]

    y_pred, y_true = cross_validate(X, y, models, splitter, pred_type="probabilities")

    models[0].assert_has_calls(mod0_expected_calls)
    models[1].assert_has_calls(mod1_expected_calls)
    assert len(y_pred) == 2
    assert y_true == [s.y_te0, s.y_te1]

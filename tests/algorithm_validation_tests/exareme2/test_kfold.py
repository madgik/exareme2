import json
from pathlib import Path

import pytest

from tests.algorithm_validation_tests.exareme2.helpers import algorithm_request
from tests.algorithm_validation_tests.exareme2.helpers import parse_response

test_case_file = Path(__file__).parent / "expected" / "test_cases_kfold.json"
with test_case_file.open() as f:
    test_cases = json.load(f)


@pytest.mark.parametrize("test_input", test_cases)
def test_kfold(test_input):
    response = algorithm_request("test_kfold", test_input)
    row_sets = parse_response(response)

    xfull = row_sets["xfull"]
    yfull = row_sets["yfull"]
    xtrain = row_sets["xtrain"]
    xtest = row_sets["xtest"]
    ytrain = row_sets["ytrain"]
    ytest = row_sets["ytest"]

    assert_disjoint_train_test(xtrain, xtest)
    assert_conservation_of_datapoints(xtrain, xtest, xfull)
    assert_disjoint_folds(xtest)

    assert_disjoint_train_test(ytrain, ytest)
    assert_conservation_of_datapoints(ytrain, ytest, yfull)
    assert_disjoint_folds(ytest)

    assert_x_y_rows_match(xtrain, ytrain)
    assert_x_y_rows_match(xtest, ytest)


def assert_disjoint_train_test(train, test):
    msg = "Train and test sets must be disjoint for all folds"
    for tr, te in zip(train, test):
        assert not (set(tr) & set(te)), msg


def assert_conservation_of_datapoints(train, test, full):
    msg = "Train and test sets must add up to whole set for all folds"
    for tr, te in zip(train, test):
        assert set(tr) | set(te) == set(full), msg


def assert_disjoint_folds(test):
    msg = "All folds must result in disjoint test sets"
    n_splits = len(test)
    for i in range(n_splits):
        for j in range(i + 1, n_splits):
            assert not (set(test[i]) & set(test[j])), msg


def assert_x_y_rows_match(xset, yset):
    msg = "X set and Y set must have identical row ids for all folds"
    for x, y in zip(xset, yset):
        assert x == y, msg

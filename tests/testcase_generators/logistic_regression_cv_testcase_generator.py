import warnings

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

from tests.testcase_generators.testcase_generator import TestCaseGenerator

warnings.filterwarnings("ignore")


class LogisticRegressionCVTestCaseGenerator(TestCaseGenerator):
    def compute_expected_output(self, input_data, input_parameters):
        y, X = input_data
        positive_class = input_parameters["positive_class"]
        n_splits = input_parameters["n_splits"]

        ybin = (y == positive_class).astype(int)

        # reject if there are fewer than 2 classes
        if len(ybin.iloc[:, 0].unique()) < 2:
            return

        [yname] = y.columns
        xnames = X.columns
        data = pd.concat([ybin, X], axis=1)
        data.dropna(inplace=True)

        # reject if class counts are less than the number of columns of X
        positive_count = sum(data[yname])
        negative_count = len(data) - positive_count
        # The  actual  number  of  columns  is larger than len(X.columns) due to
        # dummy variables. It is too much hassle to compute this here, so I just
        # catch  these  cases  during  algorithm validation testing, and replace
        # them with newly generated ones.
        # WARNING  This  is enough in the case of simple logistic regression but
        # doesn't  catch some degenerate cases with cross validation. The reason
        # is  that  the  condition  should  be  verified for each fold, which is
        # currently  a  lot  more  work.  When those cases arise they are caught
        # during testing and replaced with new ones.
        n_cols = len(X.columns) + 1
        if positive_count <= n_cols or negative_count <= n_cols:
            return

        X = data[xnames]
        ybin = data[yname]

        model = LogisticRegression(
            penalty="none",
            tol=1e-4,
            fit_intercept=True,
            solver="newton-cg",
        )

        try:
            result = cross_validate(
                model,
                X,
                ybin,
                scoring=("accuracy", "recall", "precision", "f1", "roc_auc"),
                cv=n_splits,
            )
        except Exception as ex:
            print(ex)
            return

        return {
            "accuracy": result["test_accuracy"].tolist(),
            "recall": result["test_recall"].tolist(),
            "precision": result["test_precision"].tolist(),
            "fscore": result["test_f1"].tolist(),
            "auc": result["test_roc_auc"].tolist(),
        }


if __name__ == "__main__":
    algorithm_properties_file = "exareme3/algorithms/logistic_regression_cv.json"
    expected_file = (
        "tests/algorithm_validation_tests/expected/logisticregression_cv_expected.json"
    )
    with open(algorithm_properties_file) as specs_file:
        gen = LogisticRegressionCVTestCaseGenerator(specs_file)
    with open(expected_file, "w") as file:
        gen.write_test_cases(file, num_test_cases=35)

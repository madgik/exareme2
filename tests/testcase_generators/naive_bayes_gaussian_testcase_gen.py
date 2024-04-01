import json
from collections import Counter

from sklearn.naive_bayes import GaussianNB

from exareme2.algorithms.exareme2.naive_bayes_gaussian_cv import GaussianNBAlgorithm
from tests.testcase_generators.testcase_generator import TestCaseGenerator


class GaussianNBFitTestCaseGenerator(TestCaseGenerator):
    def compute_expected_output(self, input_data, *_):
        y, X = input_data
        X = X.values
        y = y.iloc[:, 0].values.ravel()

        gnb = GaussianNB()
        gnb.fit(X, y)

        return {
            "theta": gnb.theta_.tolist(),
            "var": gnb.var_.tolist(),
            "class_count": gnb.class_count_.tolist(),
        }


class GaussianNBPredictTestCaseGenerator(TestCaseGenerator):
    def compute_expected_output(self, input_data, *_):
        y, X = input_data
        X = X.values
        y = y.iloc[:, 0].values.ravel()

        gnb = GaussianNB()
        gnb.fit(X, y)
        # It is impossible to compare predictions directly as the row order is
        # not guaranteed. Instread I compare prediction counts.
        predictions = Counter(gnb.predict(X))

        return {"predictions": predictions}


if __name__ == "__main__":
    specs = json.loads(GaussianNBAlgorithm.get_specification().json(exclude_none=True))

    gen = GaussianNBFitTestCaseGenerator(specs)
    with open("tmp.json", "w") as expected_file:
        gen.write_test_cases(expected_file, num_test_cases=5)

    gen = GaussianNBPredictTestCaseGenerator(specs)
    with open("tmp2.json", "w") as expected_file:
        gen.write_test_cases(expected_file, num_test_cases=5)

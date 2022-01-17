from pathlib import Path

from scipy.stats import pearsonr

from tests.testcase_generators.testcase_generator import TestCaseGenerator

SPECS_PATH = Path("mipengine", "algorithms", "pearson.json")
EXPECTED_PATH = Path(
    "tests",
    "prod_env_tests",
    "algorithm_validation_tests",
    "expected",
    "pearson_expected.json",
)


class PearsonTestCaseGenerator(TestCaseGenerator):
    def compute_expected_output(self, input_data, input_parameters=None):
        Y, X = input_data
        if X is None:
            X = Y

        n_obs = len(Y)

        correlation_table = {}
        correlation_table["variables"] = list(X.columns)
        pvalues_table = {}
        pvalues_table["variables"] = list(X.columns)
        for y_col in Y.columns:
            results = [pearsonr(Y[y_col], X[x_col]) for x_col in X.columns]
            correlations = [corr for corr, _ in results]
            pvalues = [pval for _, pval in results]
            correlation_table[y_col] = correlations
            pvalues_table[y_col] = pvalues

        return {
            "n_obs": n_obs,
            "correlations": correlation_table,
            "p-values": pvalues_table,
        }


if __name__ == "__main__":
    with open(SPECS_PATH) as specs_file:
        pcagen = PearsonTestCaseGenerator(specs_file)
    with open(EXPECTED_PATH, "w") as expected_file:
        pcagen.write_test_cases(expected_file)

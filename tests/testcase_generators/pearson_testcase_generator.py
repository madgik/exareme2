import warnings
from pathlib import Path

import numpy as np
from scipy import stats

from tests.testcase_generators.testcase_generator import TestCaseGenerator

SPECS_PATH = Path("exareme3", "algorithms", "pearson_correlation.json")
EXPECTED_PATH = Path(
    "tests",
    "prod_env_tests",
    "algorithm_validation_tests",
    "expected",
    "pearson_correlation_expected.json",
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
        low_ci_table = {}
        low_ci_table["variables"] = list(X.columns)
        high_ci_table = {}
        high_ci_table["variables"] = list(X.columns)
        for y_col in Y.columns:
            # correlations and p-values
            results = [stats.pearsonr(Y[y_col], X[x_col]) for x_col in X.columns]
            correlations = [corr for corr, _ in results]
            pvalues = [pval for _, pval in results]
            correlation_table[y_col] = correlations
            pvalues_table[y_col] = pvalues

            # confidence intervals
            alpha = input_parameters["alpha"]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r_z = np.arctanh(correlations)
            se = 1 / np.sqrt(n_obs - 3)
            z = stats.norm.ppf(1 - alpha / 2)
            lo_z, hi_z = r_z - z * se, r_z + z * se
            lo, hi = np.tanh((lo_z, hi_z))
            low_ci_table[y_col] = lo.tolist()
            high_ci_table[y_col] = hi.tolist()

        return {
            "n_obs": n_obs,
            "correlations": correlation_table,
            "p-values": pvalues_table,
            "low_confidence_intervals": low_ci_table,
            "high_confidence_intervals": high_ci_table,
        }


if __name__ == "__main__":
    with open(SPECS_PATH) as specs_file:
        pcagen = PearsonTestCaseGenerator(specs_file)
    with open(EXPECTED_PATH, "w") as expected_file:
        pcagen.write_test_cases(expected_file)

from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pt
from scipy import stats

from tests.testcase_generators.testcase_generator import TestCaseGenerator

SPECS_PATH = Path("mipengine", "algorithms", "paired_ttest.json")
EXPECTED_PATH = Path(
    "tests",
    "algorithm_validation_tests",
    "expected",
    "paired_ttest_expected.json",
)


class PairedTtestTestCaseGenerator(TestCaseGenerator):
    def compute_expected_output(self, input_data, input_parameters="alt_hypothesis"):
        Y, X = input_data
        alt_hyp = input_parameters["alt_hypothesis"]
        alpha = input_parameters["alpha"]
        n_obs = len(Y)
        y_name = Y.columns[0]
        x_name = X.columns[0]
        dict_to_df = {y_name: np.array(Y[y_name]), x_name: np.array(X[x_name])}
        data = pd.DataFrame(dict_to_df)
        y = data[y_name].tolist()
        x = data[x_name].tolist()

        x1_sqrd_sum = sum(X[x_name] ** 2)
        # x2_sqrd_sum = sum(Y[y_name] ** 2)
        diff_sum = sum(X[x_name] - Y[y_name])
        diff_sqrd_sum = sum((X[x_name] - Y[y_name]) ** 2)

        # standard deviation of the difference between means
        sd = np.sqrt((diff_sqrd_sum - (diff_sum**2 / n_obs)) / (n_obs - 1))

        # standard error of the difference between means
        sed = sd / np.sqrt(n_obs)
        sample_mean = diff_sum / n_obs

        # critical value
        cv = stats.t.ppf(1.0 - alpha / 2, n_obs - 1)

        ci_upper = sample_mean + cv * (sd / np.sqrt(n_obs))
        ci_lower = sample_mean - cv * (sd / np.sqrt(n_obs))
        res = stats.ttest_rel(x, y, alternative=alt_hyp, nan_policy="omit")
        cohens_d = (np.mean(X[x_name]) - np.mean(Y[y_name])) / np.sqrt(
            (np.std(x, ddof=1) ** 2 + np.std(y, ddof=1) ** 2) / 2
        )

        devel1 = sum(((X[x_name]) - np.mean(X[x_name])) ** 2)
        devel2 = sum(((Y[y_name]) - np.mean(Y[y_name])) ** 2)

        expected_out = {
            "statistic": res[0],
            "p_value": res[1],
            "df": n_obs - 1,
            "mean_diff": diff_sum,
            "se_difference": sed,
            "ci_upper": ci_upper,
            "ci_lower": ci_lower,
            "cohens_d": cohens_d,
            "devel1": devel1,
            "devel2": devel2,
            "x1_sqrd_sum": x1_sqrd_sum,
            "mean_x1": np.mean(X[x_name]),
            "sum_x1": sum(X[x_name]),
        }

        return expected_out


if __name__ == "__main__":
    with open(SPECS_PATH) as specs_file:
        paired_gen = PairedTtestTestCaseGenerator(specs_file)
    with open(EXPECTED_PATH, "w") as expected_file:
        paired_gen.write_test_cases(expected_file)

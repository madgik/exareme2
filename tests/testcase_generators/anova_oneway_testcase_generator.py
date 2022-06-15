from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pingouin import pairwise_tukey
from statsmodels.formula.api import ols
from statsmodels.stats.libqsturng import psturng

from tests.testcase_generators.testcase_generator import TestCaseGenerator

SPECS_PATH = Path("mipengine", "algorithms", "one_way_anova.json")
EXPECTED_PATH = Path(
    "tests",
    "algorithm_validation_tests",
    "expected",
    "one_way_anova_expected.json",
)


class ANOVAOneWayTestCaseGenerator(TestCaseGenerator):
    def compute_expected_output(self, input_data, input_parameters=None):
        # Get data and remove missing values
        Y, X = input_data
        n_obs = len(Y)
        y_names = Y.columns[0]
        x_names = X.columns[0]
        dict_to_df = {y_names: Y[y_names].tolist(), x_names: X[x_names].tolist()}
        data = pd.DataFrame(dict_to_df)
        n_groups = len(set(data[x_names]))

        if n_groups < 2:
            raise ValueError(
                f"Not enough enums to create test case. Variable:\033[1m '{x_names}'\033[0m"
            )
        # Anova
        formula = "{y} ~ {x}".format(y=y_names, x=x_names)
        lm = ols(formula, data=data).fit()
        aov = sm.stats.anova_lm(lm)
        result = aov.to_dict()

        # # Tukey test
        tukey = pairwise_tukey(data=data, dv=y_names, between=x_names)
        tukey_results = []
        for _, row in tukey.iterrows():
            tukey_result = dict()
            tukey_result["groupA"] = row["A"]
            tukey_result["groupB"] = row["B"]
            tukey_result["meanA"] = row["mean(A)"]
            tukey_result["meanB"] = row["mean(B)"]
            tukey_result["diff"] = row["diff"]
            tukey_result["se"] = row["se"]
            tukey_result["t_stat"] = row["T"]
            # computing pval because pingouin and statsmodels implementations
            # of pstrung do not agree
            pval = psturng(
                np.sqrt(2) * np.abs(row["T"]), n_groups, result["df"]["Residual"]
            )
            tukey_result["p_tuckey"] = float(pval)
            tukey_results.append(tukey_result)

        expected_out = dict()
        expected_out["n_obs"] = n_obs
        expected_out["df_residual"] = result["df"]["Residual"]
        expected_out["df_explained"] = result["df"][x_names]
        expected_out["ss_residual"] = result["sum_sq"]["Residual"]
        expected_out["ss_explained"] = result["sum_sq"][x_names]
        expected_out["ms_residual"] = result["mean_sq"]["Residual"]
        expected_out["ms_explained"] = result["mean_sq"][x_names]
        expected_out["p_value"] = result["PR(>F)"][x_names]
        expected_out["f_stat"] = result["F"][x_names]
        expected_out["tuckey_test"] = tukey_results

        return expected_out


if __name__ == "__main__":
    with open(SPECS_PATH) as specs_file:
        anovagen = ANOVAOneWayTestCaseGenerator(specs_file)
    with open(EXPECTED_PATH, "w") as expected_file:
        anovagen.write_test_cases(expected_file)

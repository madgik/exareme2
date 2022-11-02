import json
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
import statsmodels.api as sm
from statsmodels.formula.api import ols

from tests.testcase_generators.testcase_generator import TestCaseGenerator

SPECS_PATH = Path("mipengine", "algorithms", "anova.json")
EXPECTED_PATH = Path(
    "tests",
    "algorithm_validation_tests",
    "expected",
    "anova_twoway_expected.json",
)


class ANOVATwoWayTestCaseGenerator(TestCaseGenerator):
    def compute_expected_output(self, input_data, input_parameters):
        Y, X = input_data
        sstype = input_parameters["sstype"]
        n_obs = len(Y)

        yname = Y.columns[0]
        xname = X.columns
        if len(xname) < 2:
            return
        else:
            x1name = xname[0]
            x2name = xname[1]
        dict_to_df = {
            yname: Y[yname].tolist(),
            x1name: X[x1name].tolist(),
            x2name: X[x2name].tolist(),
        }
        data = pd.DataFrame(dict_to_df)
        n1_groups = len(set(data[x1name]))
        n2_groups = len(set(data[x2name]))

        if n1_groups < 2 or n2_groups < 2:
            return

        # The Python way
        formula = f"{yname} ~ {x1name} + {x2name} + {x1name}:{x2name}"
        model = ols(formula, data=data).fit()
        aov_res = sm.stats.anova_lm(model, typ=sstype)

        pg_anova = pg.anova(
            dv=f"{yname}", data=data, between=[f"{x1name}", f"{x2name}"], ss_type=sstype
        )
        res_pg = pg_anova.to_dict()

        aov_res["eta_sq"] = "NaN"
        aov_res["eta_sq"] = aov_res[:-1]["sum_sq"] / sum(aov_res["sum_sq"])
        mse = aov_res["sum_sq"][-1] / aov_res["df"][-1]
        aov_res["omega_sq"] = "NaN"
        aov_res["omega_sq"] = (aov_res[:-1]["sum_sq"] - (aov_res[:-1]["df"] * mse)) / (
            sum(aov_res["sum_sq"]) + mse
        )
        aov = pd.DataFrame(aov_res)
        aov_to_dict = aov.to_dict()
        dataset_x1 = pd.DataFrame()
        dataset_x1[yname] = Y[yname]
        dataset_x1[x1name] = X[x1name]
        group_stats_x1 = dataset_x1.groupby(x1name).agg(["count", "sum"])
        dataset_x2 = pd.DataFrame()
        dataset_x2[yname] = Y[yname]
        dataset_x2[x2name] = X[x2name]
        group_stats_x2 = dataset_x2.groupby(x2name).agg(["count", "sum"])
        mean_x1 = group_stats_x1[yname]["sum"] / group_stats_x1[yname]["count"]
        mean_x2 = group_stats_x2[yname]["sum"] / group_stats_x2[yname]["count"]
        mean_y = sum(Y[yname]) / n_obs
        dfx1_ssq = sum((mean_x1 - mean_y) ** 2)
        dfx2_ssq = sum((mean_x2 - mean_y) ** 2)

        expected_out = {
            "n_obs": n_obs,
            "mean_y": mean_y,
            "dfx1_ssq": dfx1_ssq,
            "dfx2_ssq": dfx2_ssq,
            "res_pg_ssq": res_pg["SS"],
            "mean_x1": mean_x1.tolist(),
            "mean_x2": mean_x2.tolist(),
            "sum_sq": aov_to_dict["sum_sq"],
            "p_value": aov_to_dict["PR(>F)"],
            "df": aov_to_dict["df"],
            "f_stat": aov_to_dict["F"],
            "eta": aov_to_dict["eta_sq"],
            "omega": aov_to_dict["omega_sq"],
        }
        return expected_out


if __name__ == "__main__":
    with open(SPECS_PATH) as specs_file:
        paired_gen = ANOVATwoWayTestCaseGenerator(specs_file)
    with open(EXPECTED_PATH, "w") as expected_file:
        paired_gen.write_test_cases(expected_file)

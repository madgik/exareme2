import statsmodels.api as sm

from mipengine.algorithms.linear_regression import LinearRegressionResult
from tests.testcase_generators.testcase_generator import TestCaseGenerator


class LinearRegressionTestCaseGenerator(TestCaseGenerator):
    def compute_expected_output(self, input_data, _):
        y, X = input_data
        X.insert(0, "Intercept", [1] * len(X))

        model = sm.OLS(y, X, missing="drop")
        fitted = model.fit()

        result = LinearRegressionResult(
            n_obs=fitted.nobs,
            df_resid=fitted.df_resid,
            df_model=fitted.df_model,
            coefficients=fitted.params.tolist(),
            std_err=fitted.bse.tolist(),
            t_stat=fitted.tvalues.tolist(),
            t_p_values=fitted.pvalues.tolist(),
            lower_ci=fitted.conf_int().to_numpy().T.tolist()[0],
            upper_ci=fitted.conf_int().to_numpy().T.tolist()[1],
            rse=fitted.mse_resid ** 0.5,
            r_squared=fitted.rsquared,
            r_squared_adjusted=fitted.rsquared_adj,
            f_stat=fitted.fvalue,
            f_p_value=fitted.f_pvalue,
        )
        return result.dict()


if __name__ == "__main__":
    with open("mipengine/algorithms/linear_regression.json") as specs_file:
        pcagen = LinearRegressionTestCaseGenerator(specs_file)
    with open("linear_regression_expected.json", "w") as expected_file:
        pcagen.write_test_cases(expected_file)

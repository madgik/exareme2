from mipengine.algorithms.algorithm import Algorithm
from mipengine.algorithms.linear_regression import LinearRegression
from mipengine.algorithms.linear_regression import LinearRegressionResult
from mipengine.algorithms.longitudinal_transformer import LongitudinalTransformer
from mipengine.algorithms.preprocessing import DummyEncoder
from mipengine.algorithms.preprocessing import relation_to_vector


class LinearRegressionLongitudinal(Algorithm, algname="linear_regression_longitudinal"):
    def get_variable_groups(self):
        xvars = self.variables.x
        yvars = self.variables.y
        xvars += ["subjectid", "visitid"]
        yvars += ["subjectid", "visitid"]
        return [xvars, yvars]

    def run(self, engine, data, metadata):
        X, y = data
        metadata: dict = metadata

        visit1 = self.algorithm_parameters["visit1"]
        visit2 = self.algorithm_parameters["visit2"]
        strategies = self.algorithm_parameters["strategies"]

        # Split strategies to X and Y part
        x_strats = {
            name: strat
            for name, strat in strategies.items()
            if name in self.variables.x
        }
        y_strats = {
            name: strat
            for name, strat in strategies.items()
            if name in self.variables.y
        }

        xlt = LongitudinalTransformer(engine, metadata, x_strats, visit1, visit2)
        X = xlt.transform(X)
        metadata = xlt.transform_metadata(metadata)

        ylt = LongitudinalTransformer(engine, metadata, y_strats, visit1, visit2)
        y = ylt.transform(y)
        metadata = ylt.transform_metadata(metadata)

        dummy_encoder = DummyEncoder(engine=engine, metadata=metadata)
        X = dummy_encoder.transform(X)

        p = len(dummy_encoder.new_varnames) - 1

        lr = LinearRegression(engine)
        lr.fit(X=X, y=y)
        y_pred = lr.predict(X)
        lr.compute_summary(
            y_test=relation_to_vector(y, engine),
            y_pred=y_pred,
            p=p,
        )

        result = LinearRegressionResult(
            dependent_var=y.columns[0],
            n_obs=lr.n_obs,
            df_resid=lr.df,
            df_model=p,
            rse=lr.rse,
            r_squared=lr.r_squared,
            r_squared_adjusted=lr.r_squared_adjusted,
            f_stat=lr.f_stat,
            f_pvalue=lr.f_p_value,
            indep_vars=dummy_encoder.new_varnames,
            coefficients=[c[0] for c in lr.coefficients],
            std_err=lr.std_err.tolist(),
            t_stats=lr.t_stat.tolist(),
            pvalues=lr.t_p_values.tolist(),
            lower_ci=lr.ci[0].tolist(),
            upper_ci=lr.ci[1].tolist(),
        )
        return result

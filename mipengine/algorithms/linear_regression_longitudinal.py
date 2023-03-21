from mipengine.algorithm_specification import AlgorithmSpecification
from mipengine.algorithm_specification import InputDataSpecification
from mipengine.algorithm_specification import InputDataSpecifications
from mipengine.algorithm_specification import InputDataStatType
from mipengine.algorithm_specification import InputDataType
from mipengine.algorithm_specification import ParameterEnumSpecification
from mipengine.algorithm_specification import ParameterEnumType
from mipengine.algorithm_specification import ParameterSpecification
from mipengine.algorithm_specification import ParameterType
from mipengine.algorithms.algorithm import Algorithm
from mipengine.algorithms.linear_regression import LinearRegression
from mipengine.algorithms.linear_regression import LinearRegressionResult
from mipengine.algorithms.longitudinal_transformer import LongitudinalTransformer
from mipengine.algorithms.preprocessing import DummyEncoder
from mipengine.algorithms.preprocessing import relation_to_vector


class LinearRegressionLongitudinal(Algorithm, algname="linear_regression_longitudinal"):
    @classmethod
    def get_specification(cls):
        return AlgorithmSpecification(
            name=cls.algname,
            desc="Linear Regression on longitudinal dataset",
            label="Linear Regression Longitudinal",
            enabled=True,
            inputdata=InputDataSpecifications(
                x=InputDataSpecification(
                    label="features",
                    desc="Features",
                    types=[InputDataType.REAL, InputDataType.INT, InputDataType.TEXT],
                    stattypes=[InputDataStatType.NUMERICAL, InputDataStatType.NOMINAL],
                    notblank=True,
                    multiple=True,
                ),
                y=InputDataSpecification(
                    label="target",
                    desc="Target variable",
                    types=[InputDataType.REAL],
                    stattypes=[InputDataStatType.NUMERICAL],
                    notblank=True,
                    multiple=False,
                ),
            ),
            parameters={
                "visit1": ParameterSpecification(
                    label="visit1",
                    desc="visit1",
                    types=[ParameterType.TEXT],
                    notblank=True,
                    multiple=False,
                    enums=ParameterEnumSpecification(
                        type=ParameterEnumType.FIXED_VAR_CDE_ENUMS, source="visitid"
                    ),
                ),
                "visit2": ParameterSpecification(
                    label="visit2",
                    desc="visit2",
                    types=[ParameterType.TEXT],
                    notblank=True,
                    multiple=False,
                    enums=ParameterEnumSpecification(
                        type=ParameterEnumType.FIXED_VAR_CDE_ENUMS, source="visitid"
                    ),
                ),
                "strategies": ParameterSpecification(
                    label="strategies",
                    desc="strategies",
                    types=[ParameterType.DICT],
                    notblank=True,
                    multiple=False,
                    dict_keys_enums=ParameterEnumSpecification(
                        type=ParameterEnumType.INPUT_VAR_NAMES, source=["x", "y"]
                    ),
                    dict_values_enums=ParameterEnumSpecification(
                        type=ParameterEnumType.LIST, source=["diff", "first", "second"]
                    ),
                ),
            },
        )

    def get_variable_groups(self):
        xvars = self.variables.x
        yvars = self.variables.y
        xvars += ["subjectid", "visitid"]
        yvars += ["subjectid", "visitid"]
        return [xvars, yvars]

    def run(self, engine):
        X, y = engine.data_model_views
        metadata: dict = self.metadata

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

        xlt = LongitudinalTransformer(engine, self.metadata, x_strats, visit1, visit2)
        X = xlt.transform(X)
        metadata = xlt.transform_metadata(metadata)

        ylt = LongitudinalTransformer(engine, self.metadata, y_strats, visit1, visit2)
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

from mipengine.algorithm_specification import AlgorithmSpecification
from mipengine.algorithm_specification import InputDataSpecification
from mipengine.algorithm_specification import InputDataSpecifications
from mipengine.algorithm_specification import InputDataStatType
from mipengine.algorithm_specification import InputDataType
from mipengine.algorithms.algorithm import Algorithm
from mipengine.algorithms.algorithm import AlgorithmDataLoader
from mipengine.algorithms.algorithm import Variables
from mipengine.algorithms.anova import AnovaTwoWay
from mipengine.algorithms.anova import AnovaTwoWayDataLoader
from mipengine.algorithms.anova_oneway import AnovaOneWayAlgorithm
from mipengine.algorithms.anova_oneway import AnovaOneWayDataLoader
from mipengine.algorithms.linear_regression import LinearRegressionAlgorithm
from mipengine.algorithms.linear_regression import LinearRegressionDataLoader
from mipengine.algorithms.linear_regression_cv import LinearRegressionCVAlgorithm
from mipengine.algorithms.linear_regression_cv import LinearRegressionCVDataLoader
from mipengine.algorithms.logistic_regression import LogisticRegressionAlgorithm
from mipengine.algorithms.logistic_regression import LogisticRegressionDataLoader
from mipengine.algorithms.logistic_regression_cv import LogisticRegressionCVAlgorithm
from mipengine.algorithms.logistic_regression_cv import LogisticRegressionCVDataLoader
from mipengine.algorithms.longitudinal_transformer import LongitudinalTransformer
from mipengine.algorithms.naive_bayes_gaussian_cv import GaussianNBAlgorithm
from mipengine.algorithms.naive_bayes_gaussian_cv import GaussianNBDataLoader

ALGNAME = "generic_longitudinal"


class LongitudinalDataLoader(AlgorithmDataLoader, algname=ALGNAME):
    def get_variable_groups(self):
        xvars = self._variables.x
        yvars = self._variables.y
        xvars += ["subjectid", "visitid"]
        yvars += ["subjectid", "visitid"]
        return [xvars, yvars]


class LongitudinalAlgorithm(Algorithm, algname=ALGNAME):
    # TODO The system forces me to define an algorithm spec. The actual spec
    # should come dynamicly from the specific selected algorithm, but it is
    # impossible to implement this here.
    @classmethod
    def get_specification(cls):
        return AlgorithmSpecification(
            name=cls.algname,
            desc="",
            label="",
            enabled=True,
            inputdata=InputDataSpecifications(
                x=InputDataSpecification(
                    label="",
                    desc="",
                    types=[InputDataType.REAL, InputDataType.INT, InputDataType.TEXT],
                    stattypes=[InputDataStatType.NUMERICAL, InputDataStatType.NOMINAL],
                    notblank=True,
                    multiple=True,
                ),
                y=InputDataSpecification(
                    label="",
                    desc="",
                    types=[InputDataType.REAL, InputDataType.INT, InputDataType.TEXT],
                    stattypes=[InputDataStatType.NUMERICAL, InputDataStatType.NOMINAL],
                    notblank=True,
                    multiple=False,
                ),
            ),
        )

    def run(self, data, metadata):
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

        lt_x = LongitudinalTransformer(self.engine, metadata, x_strats, visit1, visit2)
        X = lt_x.transform(X)
        metadata = lt_x.transform_metadata(metadata)

        lt_y = LongitudinalTransformer(self.engine, metadata, y_strats, visit1, visit2)
        y = lt_y.transform(y)
        metadata = lt_y.transform_metadata(metadata)

        alg_vars = Variables(x=X.columns, y=y.columns)  # use transformed vars
        data = (X, y)

        return (alg_vars, data, metadata)

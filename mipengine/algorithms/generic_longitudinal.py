from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from mipengine.algorithm_specification import AlgorithmSpecification
from mipengine.algorithm_specification import InputDataSpecification
from mipengine.algorithm_specification import InputDataSpecifications
from mipengine.algorithm_specification import InputDataStatType
from mipengine.algorithm_specification import InputDataType
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
from mipengine.algorithms.longitudinal_transformer import (
    LongitudinalTransformer as LongitudinalTransformerInner,
)
from mipengine.algorithms.naive_bayes_gaussian_cv import GaussianNBAlgorithm
from mipengine.algorithms.naive_bayes_gaussian_cv import GaussianNBDataLoader

if TYPE_CHECKING:
    from mipengine.controller.algorithm_execution_engine import AlgorithmExecutionEngine


@dataclass(frozen=True)
class Result:
    data: tuple
    metadata: dict


@dataclass(frozen=True)
class Variables:
    x: List[str]
    y: List[str]


# TODO: rename AlgorithmDataloader to DataLoader and use for both algorithms and transformeers..
class DataLoader:
    def __init__(self, variables: Variables):
        self._variables = variables

    def get_variable_groups(self):
        xvars = self._variables.x
        yvars = self._variables.y
        xvars += ["subjectid", "visitid"]
        yvars += ["subjectid", "visitid"]
        return [xvars, yvars]

    def get_dropna(self) -> bool:
        return True

    def get_check_min_rows(self) -> bool:
        return True

    def get_variables(self) -> Variables:
        return self._variables


@dataclass(frozen=True)
class InitializationParams:
    datasets: List[str]
    var_filters: Optional[dict] = None
    algorithm_parameters: Optional[Dict[str, Any]] = None


class LongitudinalTransformer:
    def __init__(
        self,
        initialization_params: InitializationParams,
        data_loader: DataLoader,
        engine: "AlgorithmExecutionEngine",
    ):
        self._initialization_params = initialization_params
        self._data_loader = data_loader
        self._engine = engine

    @property
    def engine(self):
        return self._engine

    @property
    def variables(self) -> Variables:
        return self._data_loader.get_variables()

    @property
    def algorithm_parameters(self) -> Dict[str, Any]:
        return self._initialization_params.algorithm_parameters

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

        lt_x = LongitudinalTransformerInner(
            self.engine, metadata, x_strats, visit1, visit2
        )
        X = lt_x.transform(X)
        metadata = lt_x.transform_metadata(metadata)

        lt_y = LongitudinalTransformerInner(
            self.engine, metadata, y_strats, visit1, visit2
        )
        y = lt_y.transform(y)
        metadata = lt_y.transform_metadata(metadata)

        data = (X, y)

        return Result(data, metadata)

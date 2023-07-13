import typing as t
from collections import Counter

from pydantic import BaseModel

from exareme2.algorithms.algorithm import Algorithm
from exareme2.algorithms.algorithm import AlgorithmDataLoader
from exareme2.algorithms.naive_bayes_gaussian_cv import GaussianNB
from exareme2.algorithms.naive_bayes_gaussian_cv import GaussianNBAlgorithm

ALGNAME_FIT = "test_nb_gaussian_fit"


class GaussianNBDataLoaderTesting_fit(AlgorithmDataLoader, algname=ALGNAME_FIT):
    def get_variable_groups(self):
        return [self._variables.x, self._variables.y]


class GaussianNBTesting_fit(Algorithm, algname=ALGNAME_FIT):
    class Result(BaseModel):
        theta: t.List[t.List[float]]
        var: t.List[t.List[float]]
        class_count: t.List[int]

    @classmethod
    def get_specification(cls):
        return GaussianNBAlgorithm.get_specification()

    def run(self, data, metadata):
        engine = self.engine
        X, y = data

        nb = GaussianNB(engine, metadata)
        nb.fit(X, y)

        theta = nb.theta.values.tolist()
        var = nb.var.values.tolist()
        class_count = nb.class_count.values.tolist()

        return self.Result(theta=theta, var=var, class_count=class_count)


ALGNAME_PRED = "test_nb_gaussian_predict"


class GaussianNBDataLoaderTesting_predict(AlgorithmDataLoader, algname=ALGNAME_PRED):
    def get_variable_groups(self):
        return [self._variables.x, self._variables.y]


class GaussianNBTesting_predict(Algorithm, algname=ALGNAME_PRED):
    class Result(BaseModel):
        predictions: t.Dict[str, int]

    @classmethod
    def get_specification(cls):
        return GaussianNBAlgorithm.get_specification()

    def run(self, data, metadata):
        engine = self.engine
        X, y = data

        nb = GaussianNB(engine, metadata)
        nb.fit(X, y)
        predictions = nb.predict(X)
        predictions = predictions.get_table_data()[1:][0]
        predictions = Counter(predictions)

        return self.Result(predictions=predictions)

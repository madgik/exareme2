import typing as t
from collections import Counter

from pydantic import BaseModel

from exareme2.algorithms.in_database.algorithm import Algorithm
from exareme2.algorithms.in_database.algorithm import AlgorithmDataLoader
from exareme2.algorithms.in_database.naive_bayes_gaussian_cv import GaussianNB
from exareme2.algorithms.in_database.naive_bayes_gaussian_cv import GaussianNBAlgorithm
from exareme2.algorithms.in_database.specifications import AlgorithmSpecification

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
        # Use the Gaussian Naive Bayes with CV specification
        # but remove the "n_splits" parameter since this is a CV specific parameter
        gaussianNB_with_cv_specification = GaussianNBAlgorithm.get_specification()
        gaussianNB_fit_specification = AlgorithmSpecification(
            name=ALGNAME_FIT,
            desc=gaussianNB_with_cv_specification.desc,
            label=gaussianNB_with_cv_specification.label,
            enabled=gaussianNB_with_cv_specification.enabled,
            inputdata=gaussianNB_with_cv_specification.inputdata,
            parameters=None,  # Parameters are not passed
        )
        return gaussianNB_fit_specification

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
        # Use the Gaussian Naive Bayes with CV specification
        # but remove the "n_splits" parameter since this is a CV specific parameter
        gaussianNB_with_cv_specification = GaussianNBAlgorithm.get_specification()
        gaussianNB_predict_specification = AlgorithmSpecification(
            name=ALGNAME_PRED,
            desc=gaussianNB_with_cv_specification.desc,
            label=gaussianNB_with_cv_specification.label,
            enabled=gaussianNB_with_cv_specification.enabled,
            inputdata=gaussianNB_with_cv_specification.inputdata,
            parameters=None,  # Parameters are not passed
        )
        return gaussianNB_predict_specification

    def run(self, data, metadata):
        engine = self.engine
        X, y = data

        nb = GaussianNB(engine, metadata)
        nb.fit(X, y)
        predictions = nb.predict(X)
        predictions = predictions.get_table_data()[1:][0]
        predictions = Counter(predictions)

        return self.Result(predictions=predictions)

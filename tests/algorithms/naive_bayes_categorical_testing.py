import typing as t
from collections import Counter

from pydantic import BaseModel

from exareme2.algorithms.algorithm import Algorithm
from exareme2.algorithms.algorithm import AlgorithmDataLoader
from exareme2.algorithms.naive_bayes_categorical_cv import CategoricalNB
from exareme2.algorithms.naive_bayes_categorical_cv import CategoricalNBAlgorithm
from exareme2.algorithms.specifications import AlgorithmSpecification

ALGNAME_FIT = "test_nb_categorical_fit"


class CategoricalNBDataLoaderTesting_fit(AlgorithmDataLoader, algname=ALGNAME_FIT):
    def get_variable_groups(self):
        return [self._variables.x, self._variables.y]


class CategoricalNBTesting_Fit(Algorithm, algname=ALGNAME_FIT):
    class Result(BaseModel):
        category_count: t.List[t.List[t.List[int]]]
        class_count: t.List[int]

    @classmethod
    def get_specification(cls):
        # Use the Categorical Naive Bayes with CV specification
        # but remove the "n_splits" parameter since this is a CV specific parameter
        categoricalNB_with_cv_specification = CategoricalNBAlgorithm.get_specification()
        categoricalNB_fit_specification = AlgorithmSpecification(
            name=ALGNAME_FIT,
            desc=categoricalNB_with_cv_specification.desc,
            label=categoricalNB_with_cv_specification.label,
            enabled=categoricalNB_with_cv_specification.enabled,
            inputdata=categoricalNB_with_cv_specification.inputdata,
            parameters=None,  # Parameters are not passed
        )
        return categoricalNB_fit_specification

    def run(self, data, metadata):
        engine = self.engine
        X, y = data

        nb = CategoricalNB(engine, metadata)
        nb.fit(X, y)

        category_count = [val.values.tolist() for val in nb.category_count.values()]
        class_count = nb.class_count.values.tolist()

        return self.Result(category_count=category_count, class_count=class_count)


ALGNAME_PRED = "test_nb_categorical_predict"


class CategoricalNBDataLoaderTesting_predict(AlgorithmDataLoader, algname=ALGNAME_PRED):
    def get_variable_groups(self):
        return [self._variables.x, self._variables.y]


class CategoricalNBTesting_predict(Algorithm, algname=ALGNAME_PRED):
    class Result(BaseModel):
        predictions: t.Dict[str, int]

    @classmethod
    def get_specification(cls):
        # Use the Categorical Naive Bayes with CV specification
        # but remove the "n_splits" parameter since this is a CV specific parameter
        categoricalNB_with_cv_specification = CategoricalNBAlgorithm.get_specification()
        categoricalNB_predict_specification = AlgorithmSpecification(
            name=ALGNAME_FIT,
            desc=categoricalNB_with_cv_specification.desc,
            label=categoricalNB_with_cv_specification.label,
            enabled=categoricalNB_with_cv_specification.enabled,
            inputdata=categoricalNB_with_cv_specification.inputdata,
            parameters=None,  # Parameters are not passed
        )
        return categoricalNB_predict_specification

    def run(self, data, metadata):
        engine = self.engine
        X, y = data

        nb = CategoricalNB(engine, metadata)
        nb.fit(X, y)
        predictions = nb.predict(X)
        predictions = predictions.get_table_data()[1:][0]
        predictions = Counter(predictions)

        return self.Result(predictions=predictions)

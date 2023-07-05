import typing as t

from pydantic import BaseModel

from mipengine.algorithms.algorithm import Algorithm
from mipengine.algorithms.algorithm import AlgorithmDataLoader
from mipengine.algorithms.crossvalidation import KFold
from mipengine.algorithms.specifications import AlgorithmSpecification
from mipengine.algorithms.specifications import InputDataSpecification
from mipengine.algorithms.specifications import InputDataSpecifications
from mipengine.algorithms.specifications import InputDataStatType
from mipengine.algorithms.specifications import InputDataType

ALGNAME = "test_kfold"


class RowSets(BaseModel):
    xfull: t.List[str]
    yfull: t.List[str]
    xtrain: t.List[t.List[str]]
    xtest: t.List[t.List[str]]
    ytrain: t.List[t.List[str]]
    ytest: t.List[t.List[str]]


class KFoldTestDataLoader(AlgorithmDataLoader, algname=ALGNAME):
    def get_variable_groups(self):
        return [self._variables.x, self._variables.y]


class KFoldTestAlgorithm(Algorithm, algname=ALGNAME):
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
        n_splits = self.algorithm_parameters["n_splits"]

        kf = KFold(self.engine, n_splits=n_splits)
        X_train, X_test, y_train, y_test = kf.split(X, y)

        xfull_rows = X.get_table_data()[0]
        yfull_rows = y.get_table_data()[0]
        xtrain_rows = [table.get_table_data()[0] for table in X_train]
        xtest_rows = [table.get_table_data()[0] for table in X_test]
        ytrain_rows = [table.get_table_data()[0] for table in y_train]
        ytest_rows = [table.get_table_data()[0] for table in y_test]

        return RowSets(
            xfull=xfull_rows,
            yfull=yfull_rows,
            xtrain=xtrain_rows,
            xtest=xtest_rows,
            ytrain=ytrain_rows,
            ytest=ytest_rows,
        )

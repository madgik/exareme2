import typing as t

from pydantic import BaseModel

from exareme2.algorithms.exareme2.algorithm import Algorithm
from exareme2.algorithms.exareme2.algorithm import AlgorithmDataLoader
from exareme2.algorithms.exareme2.crossvalidation import KFold

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

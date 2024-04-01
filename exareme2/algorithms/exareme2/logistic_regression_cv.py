import statistics as stats
from typing import List
from typing import Optional

import sklearn.metrics as skm
from pydantic import BaseModel

from exareme2.algorithms.exareme2.algorithm import Algorithm
from exareme2.algorithms.exareme2.algorithm import AlgorithmDataLoader
from exareme2.algorithms.exareme2.crossvalidation import KFold
from exareme2.algorithms.exareme2.crossvalidation import cross_validate
from exareme2.algorithms.exareme2.logistic_regression import LogisticRegression
from exareme2.algorithms.exareme2.metrics import compute_classification_metrics
from exareme2.algorithms.exareme2.metrics import confusion_matrix_binary
from exareme2.algorithms.exareme2.metrics import roc_curve
from exareme2.algorithms.exareme2.preprocessing import DummyEncoder
from exareme2.algorithms.exareme2.preprocessing import LabelBinarizer
from exareme2.algorithms.specifications import AlgorithmName

ALGORITHM_NAME = AlgorithmName.LOGISTIC_REGRESSION_CV


class LogisticRegressionCVDataLoader(AlgorithmDataLoader, algname=ALGORITHM_NAME):
    def get_variable_groups(self):
        return [self._variables.x, self._variables.y]


class LogisticRegressionCVAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self, data, metadata):
        X, y = data

        positive_class = self.algorithm_parameters["positive_class"]
        n_splits = self.algorithm_parameters["n_splits"]

        # Dummy encode categorical variables
        dummy_encoder = DummyEncoder(engine=self.engine, metadata=metadata)
        X = dummy_encoder.transform(X)

        # Binarize `y` by mapping positive_class to 1 and everything else to 0
        ybin = LabelBinarizer(self.engine, positive_class).transform(y)

        # Perform cross-validation
        kf = KFold(self.engine, n_splits=n_splits)
        models = [LogisticRegression(self.engine) for _ in range(n_splits)]
        probas, y_true = cross_validate(X, ybin, models, kf, pred_type="probabilities")

        # Patrial and total confusion matrices
        confmats = [
            confusion_matrix_binary(self.engine, ytrue, proba)
            for ytrue, proba in zip(y_true, probas)
        ]
        total_confmat = sum(confmats)
        tn, fp, fn, tp = total_confmat.ravel()
        confmat = ConfusionMatrix(tn=tn, fp=fp, fn=fn, tp=tp)

        # Classification metrics
        metrics = [compute_classification_metrics(confmat) for confmat in confmats]
        n_obs_train = [model.nobs_train for model in models]
        summary = make_classification_metrics_summary(n_splits, n_obs_train, metrics)

        # ROC curves
        roc_curves = [
            roc_curve(self.engine, ytrue, proba) for ytrue, proba in zip(y_true, probas)
        ]
        aucs = [skm.auc(x=fpr, y=tpr) for tpr, fpr in roc_curves]
        roc_curves_result = [
            ROCCurve(name=f"fold_{i}", tpr=tpr, fpr=fpr, auc=auc)
            for (tpr, fpr), auc, i in zip(roc_curves, aucs, range(n_splits))
        ]

        return CVLogisticRegressionResult(
            dependent_var=y.columns[0],
            indep_vars=X.columns,
            summary=summary,
            confusion_matrix=confmat,
            roc_curves=roc_curves_result,
        )


def make_classification_metrics_summary(n_splits, n_obs, metrics):
    row_names = [f"fold_{i}" for i in range(1, n_splits + 1)] + ["average", "stdev"]
    accuracy, precision, recall, fscore = zip(*metrics)
    accuracy += (stats.mean(accuracy), stats.stdev(accuracy))
    precision += (stats.mean(precision), stats.stdev(precision))
    recall += (stats.mean(recall), stats.stdev(recall))
    fscore += (stats.mean(fscore), stats.stdev(fscore))
    return CVClassificationSummary(
        row_names=row_names,
        n_obs=n_obs + [None, None],  # we don't compute average & stderr for n_obs
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        fscore=fscore,
    )


class ConfusionMatrix(BaseModel):
    tp: int
    fp: int
    tn: int
    fn: int

    def __add__(self, other):
        return ConfusionMatrix(
            tp=self.tp + other.tp,
            fp=self.fp + other.fp,
            tn=self.tn + other.tn,
            fn=self.fn + other.fn,
        )


class CVClassificationSummary(BaseModel):
    row_names: List[str]
    n_obs: List[Optional[int]]
    accuracy: List[float]
    precision: List[float]
    recall: List[float]
    fscore: List[float]


class ROCCurve(BaseModel):
    name: str
    tpr: List[float]
    fpr: List[float]
    auc: float


class CVLogisticRegressionResult(BaseModel):
    dependent_var: str
    indep_vars: List[str]
    summary: CVClassificationSummary
    confusion_matrix: ConfusionMatrix
    roc_curves: List[ROCCurve]

import statistics as stats
from typing import List
from typing import Optional

import sklearn.metrics as skm
from pydantic import BaseModel

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
from mipengine.algorithms.logistic_regression import LogisticRegression
from mipengine.algorithms.metrics import compute_classification_metrics
from mipengine.algorithms.metrics import confusion_matrix
from mipengine.algorithms.metrics import roc_curve
from mipengine.algorithms.preprocessing import DummyEncoder
from mipengine.algorithms.preprocessing import KFold
from mipengine.algorithms.preprocessing import LabelBinarizer


class LogisticRegressionCVAlgorithm(Algorithm, algname="logistic_regression_cv"):
    @classmethod
    def get_specification(cls):
        return AlgorithmSpecification(
            name=cls.algname,
            desc="Logistic Regression Cross-validation",
            label="Logistic Regression Cross-validation",
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
                    desc="Target",
                    types=[InputDataType.INT, InputDataType.TEXT],
                    stattypes=[InputDataStatType.NOMINAL],
                    notblank=True,
                    multiple=False,
                ),
            ),
            parameters={
                "positive_class": ParameterSpecification(
                    label="Positive class",
                    desc="Positive class of y. All other classes are considered negative.",
                    types=[ParameterType.TEXT, ParameterType.INT],
                    notblank=True,
                    multiple=False,
                    enums=ParameterEnumSpecification(
                        type=ParameterEnumType.INPUT_VAR_CDE_ENUMS,
                        source="y",
                    ),
                ),
                "n_splits": ParameterSpecification(
                    label="Number of splits",
                    desc="Number of splits",
                    types=[ParameterType.INT],
                    notblank=True,
                    multiple=False,
                    default=5,
                    min=2,
                    max=20,
                ),
            },
        )

    def get_variable_groups(self):
        return [self.variables.x, self.variables.y]

    def run(self, engine):
        X, y = engine.data_model_views

        positive_class = self.algorithm_parameters["positive_class"]
        n_splits = self.algorithm_parameters["n_splits"]

        # Dummy encode categorical variables
        dummy_encoder = DummyEncoder(engine=engine, metadata=self.metadata)
        X = dummy_encoder.transform(X)

        # Binarize `y` by mapping positive_class to 1 and everything else to 0
        ybin = LabelBinarizer(engine, positive_class).transform(y)

        # Split datasets according to k-fold CV
        kf = KFold(engine, n_splits=n_splits)
        X_train, X_test, y_train, y_test = kf.split(X, ybin)

        # Create models
        models = [LogisticRegression(engine) for _ in range(n_splits)]

        # Train models
        for model, X, y in zip(models, X_train, y_train):
            model.fit(X=X, y=y)

        # Compute prediction probabilities
        probas = [model.predict_proba(X) for model, X in zip(models, X_test)]

        # Patrial and total confusion matrices
        confmats = [
            confusion_matrix(engine, ytrue, proba)
            for ytrue, proba in zip(y_test, probas)
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
            roc_curve(engine, ytrue, proba) for ytrue, proba in zip(y_test, probas)
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

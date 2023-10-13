import sklearn.metrics as skm

from exareme2.algorithms.in_database.algorithm import Algorithm
from exareme2.algorithms.in_database.algorithm import AlgorithmDataLoader
from exareme2.algorithms.in_database.crossvalidation import KFold
from exareme2.algorithms.in_database.crossvalidation import cross_validate
from exareme2.algorithms.in_database.fedaverage import fed_average
from exareme2.algorithms.in_database.helpers import get_transfer_data
from exareme2.algorithms.in_database.helpers import sum_secure_transfers
from exareme2.algorithms.in_database.logistic_regression import LogisticRegression
from exareme2.algorithms.in_database.logistic_regression_cv import ConfusionMatrix
from exareme2.algorithms.in_database.logistic_regression_cv import (
    CVLogisticRegressionResult,
)
from exareme2.algorithms.in_database.logistic_regression_cv import (
    LogisticRegressionCVAlgorithm,
)
from exareme2.algorithms.in_database.logistic_regression_cv import ROCCurve
from exareme2.algorithms.in_database.logistic_regression_cv import (
    make_classification_metrics_summary,
)
from exareme2.algorithms.in_database.metrics import compute_classification_metrics
from exareme2.algorithms.in_database.metrics import confusion_matrix_binary
from exareme2.algorithms.in_database.metrics import roc_curve
from exareme2.algorithms.in_database.preprocessing import DummyEncoder
from exareme2.algorithms.in_database.preprocessing import LabelBinarizer
from exareme2.algorithms.in_database.specifications import AlgorithmName
from exareme2.algorithms.in_database.specifications import AlgorithmSpecification
from exareme2.algorithms.in_database.udfgen import relation
from exareme2.algorithms.in_database.udfgen import secure_transfer
from exareme2.algorithms.in_database.udfgen import udf

ALGORITHM_NAME = AlgorithmName.LOGISTIC_REGRESSION_CV_FEDAVERAGE


class LogRegCVFedAverageDataLoader(AlgorithmDataLoader, algname=ALGORITHM_NAME):
    def get_variable_groups(self):
        return [self._variables.x, self._variables.y]


class LogRegCVFedAverageAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    @classmethod
    def get_specification(cls):
        # Use the LR with CV specification but change the name
        LR_with_cv_specification = LogisticRegressionCVAlgorithm.get_specification()
        LR_with_cv_fedavg = AlgorithmSpecification(
            name=ALGORITHM_NAME,
            desc=LR_with_cv_specification.desc,
            label=LR_with_cv_specification.label,
            enabled=LR_with_cv_specification.enabled,
            inputdata=LR_with_cv_specification.inputdata,
            parameters=LR_with_cv_specification.parameters,
        )
        return LR_with_cv_fedavg

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
        models = [LogisticRegressionFedAverage(self.engine) for _ in range(n_splits)]
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


class LogisticRegressionFedAverage(LogisticRegression):
    """
    Logistic regression model with fed_average strategy

    Federated logistic regression version where the `fit` method is implemented
    using the fed_average strategy, i.e. the model is fitted independently in
    each local node and the global model is computed by averaging the model
    parameters, namely the coefficients.
    """

    # General notes of fed_average algorithms
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # The `fit` method in every fed_average model should compute the model
    # parameters locally using a SoA library (here sklearn) and then call
    # `fed_average` to compute the global model.
    #
    # `fed_average` needs the number of local nodes `num_local_nodes` which is
    # provided by the `engine` instance.
    #
    # Other parameters, besides the ones to be averaged, can be computed
    # locally but need to be placed in a different object (see `other_params`
    # below).
    def __init__(self, engine):
        self.num_local_nodes = engine.num_local_nodes
        super().__init__(engine)

    def fit(self, X, y):
        params_to_average, other_params = self.local_run(
            func=self._fit_local,
            keyword_args={"X": X, "y": y},
            share_to_global=[True, True],
        )
        averaged_params_table = self.global_run(
            func=fed_average,
            keyword_args=dict(
                params=params_to_average, num_local_nodes=self.num_local_nodes
            ),
        )
        other_params_table = self.global_run(
            func=sum_secure_transfers,
            keyword_args=dict(loctransf=other_params),
        )
        averaged_params = get_transfer_data(averaged_params_table)
        other_params = get_transfer_data(other_params_table)
        self.coeff = averaged_params["coef"]
        self.nobs_train = other_params["nobs_train"]

    @staticmethod
    @udf(
        X=relation(),
        y=relation(),
        return_type=[secure_transfer(sum_op=True), secure_transfer(sum_op=True)],
    )
    def _fit_local(X, y):
        from sklearn.linear_model import LogisticRegression as LogRegSK

        lr = LogRegSK(fit_intercept=False, penalty=None, solver="newton-cg")
        lr.fit(X, y)

        params_to_average = {}  # model parameters to average
        params_to_average["coef"] = {
            "data": lr.coef_.squeeze().tolist(),
            "operation": "sum",
            "type": "float",
        }
        other_params = {}  # other quantities not meant to be averaged
        other_params["nobs_train"] = {
            "data": len(y),
            "operation": "sum",
            "type": "int",
        }
        return params_to_average, other_params

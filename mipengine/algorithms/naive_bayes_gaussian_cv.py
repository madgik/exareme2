import typing as t

import pandas as pd
from pydantic import BaseModel

from mipengine import DType
from mipengine.algorithms.algorithm import Algorithm
from mipengine.algorithms.algorithm import AlgorithmDataLoader
from mipengine.algorithms.helpers import get_transfer_data
from mipengine.algorithms.metrics import confusion_matrix_multiclass
from mipengine.algorithms.metrics import multiclass_classification_metrics
from mipengine.algorithms.metrics import multiclass_classification_summary
from mipengine.algorithms.preprocessing import KFold
from mipengine.udfgen import DEFERRED
from mipengine.udfgen import literal
from mipengine.udfgen import relation
from mipengine.udfgen import secure_transfer
from mipengine.udfgen import transfer
from mipengine.udfgen import udf

ALGORITHM_NAME = "naive_bayes_gaussian_cv"


class ConfusionMatrix(BaseModel):
    """Multiclass confusion matrix model

    Each row of the matrix represents the instances in an actual class while
    each column represents the instances in a predicted class.

    Attributes
    ----------
    data
        Confusion matrix data in row major order
    labels
        Labels of the classes used in classification
    """

    data: t.List[t.List[int]]
    labels: t.List[str]


class MulticlassClassificationSummary(BaseModel):
    """Multiclass classification summary model

    In cross validated multiclass classification, the accuracy, precision,
    recall and fscore are computed for every class, for every fold. The number
    of observations, n_obs, is different for every fold, but doesn't depend on
    the class.

    This produces a hierarchical table. E.g. for two classes cl1, cl2 the table
    has the following form.

    |      | accuracy  | precision |  recall   |  fscore   |       |
    | fold | cl1 | cl2 | cl1 | cl2 | cl1 | cl2 | cl1 | cl2 | n_obs |
    |------+-----------+-----------+-----------+-----------|-------|
    |    1 | ..  | ..  | ..  | ..  | ..  | ..  | ..  | ..  |  ..   |
    |    2 | ..  | ..  | ..  | ..  | ..  | ..  | ..  | ..  |  ..   |

    This table is represented as a collection of mappings. For the hierarchical
    quantities these mappings are nested and have the form
        {"accuracy": {"cl1": ..., "cl2": ...}, ...}
    """

    accuracy: t.Dict[str, t.Dict[str, float]]
    precision: t.Dict[str, t.Dict[str, float]]
    recall: t.Dict[str, t.Dict[str, float]]
    fscore: t.Dict[str, t.Dict[str, float]]
    n_obs: t.Dict[str, int]


class NBResult(BaseModel):
    confusion_matrix: ConfusionMatrix
    classification_summary: MulticlassClassificationSummary


def make_naive_bayes_result(confmat, labels, summary) -> NBResult:
    confmat = ConfusionMatrix(data=confmat.tolist(), labels=labels)
    summary = MulticlassClassificationSummary(**summary)
    result = NBResult(confusion_matrix=confmat, classification_summary=summary)
    return result


class GaussianNBDataLoader(AlgorithmDataLoader, algname=ALGORITHM_NAME):
    def get_variable_groups(self):
        return [self._variables.x, self._variables.y]


class GaussianNBAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self, data, metadata):
        engine = self.engine
        X, y = data
        n_splits = self.algorithm_parameters["n_splits"]

        X_train, X_test, y_train, y_test = KFold(engine, n_splits=n_splits).split(X, y)

        models = [GaussianNB(engine, metadata) for _ in range(n_splits)]

        for model, X, y in zip(models, X_train, y_train):
            model.fit(X=X, y=y)

        probas = [model.predict_proba(X) for model, X in zip(models, X_test)]

        labels = list(models[0].classes)
        confmats = [
            confusion_matrix_multiclass(engine, ytrue, proba, labels)
            for ytrue, proba in zip(y_test, probas)
        ]
        total_confmat = sum(confmats)

        metrics = [multiclass_classification_metrics(confmat) for confmat in confmats]
        n_obs = [model.class_count.sum() for model in models]
        summary = multiclass_classification_summary(metrics, labels, n_obs)

        return make_naive_bayes_result(total_confmat, labels, summary)


class GaussianNB:
    def __init__(self, engine, metadata):
        self.local_run = engine.run_udf_on_local_nodes
        self.global_run = engine.run_udf_on_global_node
        self.metadata = metadata

    def fit(self, X, y):
        yvar = y.columns[0]
        categories = self.metadata[yvar]["enumerations"]
        # sort to match sklearn's results
        categories = {key: categories[key] for key in sorted(categories)}
        self.categories = categories

        values, names = self.local_run(
            func=self._fit_local,
            keyword_args={"X": X, "y": y, "categories": categories},
            share_to_global=[True, True],
        )
        global_result = self.global_run(
            func=self._fit_global,
            keyword_args={"values": values, "names": names},
        )
        global_result = get_transfer_data(global_result)
        theta = global_result["means"]  # means is called theta in sklearn
        var = global_result["var"]
        index = global_result["index"]
        columns = global_result["columns"]
        class_count = global_result["class_count"]
        classes = global_result["classes"]
        self.classes = classes
        self.theta = pd.DataFrame(data=theta, index=index, columns=columns)
        self.var = pd.DataFrame(data=var, index=index, columns=columns)
        self.class_count = pd.Series(data=class_count, index=classes)
        self.class_prior = self.class_count / self.class_count.sum()

    @staticmethod
    @udf(
        X=relation(),
        y=relation(),
        categories=literal(),
        return_type=[secure_transfer(sum_op=True), transfer()],
    )
    def _fit_local(X, y, categories):
        import pandas as pd

        data = X
        yvar = y.columns[0]
        data[yvar] = pd.Categorical(y[yvar], categories=categories)

        def sum_sq(x):
            return sum(x**2)

        agg = data.groupby(by=yvar).agg(["count", "sum", sum_sq])
        agg = agg.swaplevel(axis=1)

        counts = agg.xs("count", axis=1)
        sums = agg.xs("sum", axis=1)
        sums_sq = agg.xs("sum_sq", axis=1)

        values = {}
        values["counts"] = {
            "data": counts.values.tolist(),
            "type": "int",
            "operation": "sum",
        }
        values["sums"] = {
            "data": sums.values.tolist(),
            "type": "float",
            "operation": "sum",
        }
        values["sums_sq"] = {
            "data": sums_sq.values.tolist(),
            "type": "int",
            "operation": "sum",
        }

        names = {}
        names["index"] = counts.index.tolist()
        names["columns"] = counts.columns.tolist()

        return values, names

    @staticmethod
    @udf(
        values=secure_transfer(sum_op=True),
        names=transfer(),
        return_type=transfer(),
    )
    def _fit_global(values, names):
        import pandas as pd

        # Portion of the largest variance of all features that is added to
        # variances for calculation stability.
        VAR_SMOOTHING = 1e-9

        index = names["index"]
        columns = names["columns"]

        counts = pd.DataFrame(data=values["counts"], index=index, columns=columns)
        sums = pd.DataFrame(data=values["sums"], index=index, columns=columns)
        sums_sq = pd.DataFrame(data=values["sums_sq"], index=index, columns=columns)

        # When count=0 then mean and var are undefined therefor I drop those rows.
        zero_indices = (counts == 0).all(axis=1)
        counts = counts[~zero_indices]
        sums = sums[~zero_indices]
        sums_sq = sums_sq[~zero_indices]

        means = sums / counts
        var = (sums_sq - 2 * means * sums + counts * (means**2)) / counts

        # When count=1 then var=0 which leads to undefined predictions due to
        # the prediction probability being a delta function. Clipping to a very
        # small value approximates the delta function with a very narrow
        # Gaussian. The epsilon computation matches sklearn's.
        epsilon = VAR_SMOOTHING * var.values.max()
        var.clip(lower=epsilon, inplace=True)

        class_count = counts.iloc[:, 0]
        classes = class_count.index

        result = {}
        result["columns"] = names["columns"]
        result["index"] = counts.index.tolist()
        result["means"] = means.values.tolist()
        result["var"] = var.values.tolist()
        result["class_count"] = class_count.tolist()
        result["classes"] = classes.tolist()
        return result

    def predict_proba(self, X):
        if not (hasattr(self, "theta") and hasattr(self, "var")):
            cls_name = self.__class__.__name__
            msg = f"{cls_name} is not fitted yet. Call 'fit' with "
            msg += "appropriate arguments before using 'predict'."
            raise ValueError(msg)

        classes = self.class_count.index.tolist()
        columns = [f"col{i}" for i, _ in enumerate(classes)]
        output_schema = [("row_id", DType.INT)]
        output_schema += [(colname, DType.FLOAT) for colname in columns]
        return self.local_run(
            self._predict_proba_local,
            keyword_args={
                "X": X,
                "theta": self.theta.values.tolist(),
                "var": self.var.values.tolist(),
                "class_count": self.class_count.values.tolist(),
                "columns": columns,
            },
            output_schema=output_schema,
        )

    @staticmethod
    @udf(
        X=relation(),
        theta=literal(),
        var=literal(),
        class_count=literal(),
        columns=literal(),
        return_type=relation(schema=DEFERRED),
    )
    def _predict_proba_local(X, theta, var, class_count, columns):
        import numpy as np
        import pandas as pd
        from scipy import stats

        index = X.index
        X = X.values

        theta = np.array(theta)
        var = np.array(var)
        class_count = np.array(class_count)

        # Each factor is a normal distribution
        #     P(x_i | y) ~ N(θ_y, σ_y)
        factors = stats.norm.pdf(X[:, np.newaxis, :], loc=theta, scale=var**0.5)

        # The total likelihood is given by the product
        #     Π_i P(x_i | y)
        likelihood = factors.prod(axis=2)

        # The prior P(y) is computed by counting
        prior = class_count / class_count.sum()

        # The posterior is given by Bayes' theorem
        #     P(y | x_1, ..., x_n) ~ P(y) Π_i P(x_i | y)
        unnormalized_post = prior * likelihood

        # Finally normalize posterior
        posterior = unnormalized_post / unnormalized_post.sum(axis=1)[:, np.newaxis]

        proba = pd.DataFrame(data=posterior, index=index, columns=columns)
        return proba

    def predict(self, X):
        probas = self.predict_proba(X)
        return self.local_run(
            self._predict_local,
            keyword_args={"probas": probas, "classes": self.classes},
        )

    @staticmethod
    @udf(
        probas=relation(),
        classes=literal(),
        return_type=relation(schema=[("row_id", DType.INT), ("pred", DType.STR)]),
    )
    def _predict_local(probas, classes):
        import numpy as np
        import pandas as pd

        index = probas.index
        probas = probas.values
        classes = np.array(classes)

        predictions = classes[np.argmax(probas, axis=1)]

        predictions = pd.DataFrame(data=predictions, index=index, columns=["pred"])
        return predictions

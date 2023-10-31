import pandas as pd

from exareme2 import DType
from exareme2.algorithms.in_database.algorithm import Algorithm
from exareme2.algorithms.in_database.algorithm import AlgorithmDataLoader
from exareme2.algorithms.in_database.crossvalidation import KFold
from exareme2.algorithms.in_database.crossvalidation import cross_validate
from exareme2.algorithms.in_database.helpers import get_transfer_data
from exareme2.algorithms.in_database.metrics import confusion_matrix_multiclass
from exareme2.algorithms.in_database.metrics import multiclass_classification_metrics
from exareme2.algorithms.in_database.metrics import multiclass_classification_summary
from exareme2.algorithms.in_database.naive_bayes_gaussian_cv import (
    make_naive_bayes_result,
)
from exareme2.algorithms.in_database.specifications import AlgorithmName
from exareme2.algorithms.in_database.udfgen import DEFERRED
from exareme2.algorithms.in_database.udfgen import literal
from exareme2.algorithms.in_database.udfgen import relation
from exareme2.algorithms.in_database.udfgen import secure_transfer
from exareme2.algorithms.in_database.udfgen import transfer
from exareme2.algorithms.in_database.udfgen import udf

ALGORITHM_NAME = AlgorithmName.NAIVE_BAYES_CATEGORICAL_CV


class CategoricalNBDataLoader(AlgorithmDataLoader, algname=ALGORITHM_NAME):
    def get_variable_groups(self):
        return [self._variables.x, self._variables.y]


class CategoricalNBAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self, data, metadata):
        engine = self.engine
        X, y = data
        n_splits = self.algorithm_parameters["n_splits"]

        # Perform cross-validation
        kf = KFold(self.engine, n_splits=n_splits)
        models = [CategoricalNB(engine, metadata) for _ in range(n_splits)]
        probas, y_true = cross_validate(X, y, models, kf, pred_type="probabilities")

        labels = list(models[0].classes)
        confmats = [
            confusion_matrix_multiclass(engine, y_t, proba, labels)
            for y_t, proba in zip(y_true, probas)
        ]
        total_confmat = sum(confmats)

        metrics = [multiclass_classification_metrics(confmat) for confmat in confmats]
        n_obs = [model.class_count.sum() for model in models]
        summary = multiclass_classification_summary(metrics, labels, n_obs)

        return make_naive_bayes_result(total_confmat, labels, summary)


class CategoricalNB:
    def __init__(self, engine, metadata):
        self.local_run = engine.run_udf_on_local_nodes
        self.global_run = engine.run_udf_on_global_node
        self.metadata = metadata

    def fit(self, X, y):
        vars = list(X.columns) + list(y.columns)
        categories = {var: self.metadata[var]["enumerations"].keys() for var in vars}
        # Sort to match sklearn's results
        sorted_cats = {var: list(sorted(cats)) for var, cats in categories.items()}
        self.categories = categories

        values, names = self.local_run(
            func=self._fit_local,
            keyword_args={"X": X, "y": y, "categories": sorted_cats},
            share_to_global=[True, True],
        )
        global_transf = self.global_run(
            func=self._fit_global,
            keyword_args={"values": values, "names": names},
        )
        global_transf = get_transfer_data(global_transf)

        counts = global_transf["counts"]
        indices = global_transf["indices"]

        yvar = y.columns[0]
        category_count = {
            var: pd.Series(
                count,
                index=pd.MultiIndex.from_tuples(indices[var], names=[yvar, var]),
            ).unstack()
            for count, (var, index) in zip(counts, indices.items())
        }

        class_count = global_transf["class_count"]
        class_index = global_transf["class_index"]
        class_count = pd.Series(class_count, index=class_index)

        # Remove rows where class_count == 0
        self.category_count = {
            var: cc[class_count != 0] for var, cc in category_count.items()
        }
        self.class_count = class_count[class_count != 0]
        self.classes = self.class_count.index.tolist()

    @staticmethod
    @udf(
        X=relation(),
        y=relation(),
        categories=literal(),
        return_type=[secure_transfer(sum_op=True), transfer()],
    )
    def _fit_local(X, y, categories):
        import pandas as pd

        xvars = X.columns
        yvar = y.columns[0]

        # Create single dataframe with all categorical variables
        df = pd.DataFrame()
        df[yvar] = pd.Categorical(y[yvar], categories=categories[yvar])
        for xvar in xvars:
            df[xvar] = pd.Categorical(X[xvar], categories=categories[xvar])

        count_dfs = {var: df.groupby([yvar, var]).size() for var in xvars}
        counts = [d.values.tolist() for d in count_dfs.values()]
        indices = {var: d.index.tolist() for var, d in count_dfs.items()}

        # compute class counts
        agg = df.groupby(by=yvar).agg(["count"])
        agg = agg.swaplevel(axis=1)
        class_count = agg.xs("count", axis=1)

        stransf = {}
        stransf["counts"] = {
            "data": counts,
            "type": "int",
            "operation": "sum",
        }
        stransf["class_count"] = {
            "data": class_count.values.tolist(),
            "type": "int",
            "operation": "sum",
        }

        transf = {}
        transf["indices"] = indices
        transf["class_index"] = class_count.index.tolist()

        return stransf, transf

    @staticmethod
    @udf(
        values=secure_transfer(sum_op=True),
        names=transfer(),
        return_type=transfer(),
    )
    def _fit_global(values, names):
        index = names["class_index"]
        class_count = values["class_count"]

        counts = pd.DataFrame(data=class_count, index=index)

        class_count = counts.iloc[:, 0]
        classes = class_count.index

        result = {}
        result["indices"] = names["indices"]
        result["class_index"] = index
        result["class_count"] = class_count.tolist()
        result["classes"] = classes.tolist()
        result["counts"] = values["counts"]
        return result

    def predict_proba(self, X):
        if not hasattr(self, "category_count"):
            cls_name = self.__class__.__name__
            msg = f"{cls_name} is not fitted yet. Call 'fit' with "
            msg += "appropriate arguments before using 'predict'."
            raise ValueError(msg)

        classes = self.class_count.index.tolist()
        columns = [f"col{i}" for i, _ in enumerate(classes)]
        output_schema = [("row_id", DType.INT)]
        output_schema += [(colname, DType.FLOAT) for colname in columns]
        categories = [
            sorted(self.metadata[key]["enumerations"].keys())
            for key in self.category_count
        ]
        return self.local_run(
            self._predict_proba_local,
            keyword_args={
                "X": X,
                "category_count": [
                    cc.values.tolist() for cc in self.category_count.values()
                ],
                "class_count": self.class_count.values.tolist(),
                "categories": categories,
                "columns": columns,
            },
            output_schema=output_schema,
        )

    @staticmethod
    @udf(
        X=relation(),
        category_count=literal(),
        class_count=literal(),
        categories=literal(),
        columns=literal(),
        return_type=relation(schema=DEFERRED),
    )
    def _predict_proba_local(X, category_count, class_count, categories, columns):
        import numpy as np
        from sklearn.preprocessing import OrdinalEncoder

        ALPHA = 1

        index = X.index
        category_count = [np.array(cc) for cc in category_count]
        class_count = np.array(class_count)

        # It is convevient to transform X with an ordinal encoder to be able to
        # use X's values as indices of the category counts. This is also how
        # sklearn implements CategoricalNB.
        X = OrdinalEncoder(categories=categories, dtype=int).fit_transform(X)

        # n_feat is the 3-dimensional tensor
        #     N_tic = |{j ∈ J | x_ij = t, y_j = c}|
        # where J = {1, ..., n_obs}.
        n_feat = np.stack([cc[:, xi] for cc, xi in zip(category_count, X.T)])

        # n_class is initially the 1-dimensional tensor
        #     N_c = |{j ∈ J | y_j = c}|
        # where J = {1, ..., n_obs}.
        # Then it is augmented to a 3-dimensional tensor by adding new axes.
        n_class = class_count[np.newaxis, :, np.newaxis]

        # n_cat is initially a 1-dimensional tensor with the number of
        # categories for each feature. Then it is augmented to a 3-dimensional
        # tensor by adding new axes.
        n_cat = np.array([len(cat) for cat in categories])[:, np.newaxis, np.newaxis]

        # Each factor corresponds to
        #     P(x_i = t | y = c; α)
        factors = (n_feat + ALPHA) / (n_class + ALPHA * n_cat)

        # The total likelihood is given by the product
        #     Π_i P(x_i | y)
        likelihood = factors.prod(axis=0).T

        # The prior P(y) is computed by counting
        prior = class_count / sum(class_count)

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

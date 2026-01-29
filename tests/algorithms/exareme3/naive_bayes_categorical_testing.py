from collections import Counter
from typing import Dict
from typing import List

from pydantic import BaseModel

from exaflow.algorithms.exareme3.naive_bayes_categorical_model import CategoricalNB
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.worker_communication import BadUserInput

ALGNAME_FIT = "test_nb_categorical_fit"
ALGNAME_PRED = "test_nb_categorical_predict"


def _sorted_categories(metadata: dict, variables: List[str]) -> Dict[str, List[str]]:
    return {
        var: list(sorted(metadata[var]["enumerations"].keys())) for var in variables
    }


class CategoricalNBTestingFit(Algorithm, algname=ALGNAME_FIT):
    class Result(BaseModel):
        category_count: List[List[List[int]]]
        class_count: List[int]

    def run(self):
        if not self.inputdata.y or not self.inputdata.x:
            raise BadUserInput("Naive Bayes categorical fit requires X and y.")

        y_var = self.inputdata.y[0]
        x_vars = list(self.inputdata.x)
        categories = _sorted_categories(self.metadata, x_vars + [y_var])

        udf_results = self.run_local_udf(
            func=categorical_nb_fit_udf,
            kw_args={
                "y_var": y_var,
                "x_vars": x_vars,
                "categories": categories,
            },
        )

        stats = udf_results[0]
        return self.Result(
            category_count=stats["category_count"],
            class_count=stats["class_count"],
        )


class CategoricalNBTestingPredict(Algorithm, algname=ALGNAME_PRED):
    class Result(BaseModel):
        predictions: Dict[str, int]

    def run(self):
        if not self.inputdata.y or not self.inputdata.x:
            raise BadUserInput("Naive Bayes categorical predict requires X and y.")

        y_var = self.inputdata.y[0]
        x_vars = list(self.inputdata.x)
        categories = _sorted_categories(self.metadata, x_vars + [y_var])

        udf_results = self.run_local_udf(
            func=categorical_nb_predict_udf,
            kw_args={
                "y_var": y_var,
                "x_vars": x_vars,
                "categories": categories,
            },
        )

        total = Counter()
        for worker_res in udf_results:
            total.update(worker_res["predictions"])

        return self.Result(predictions=dict(total))


def _prepare_dataframe(data, x_vars, y_var, categories):
    import pandas as pd

    cols = list(dict.fromkeys(list(x_vars) + [y_var]))
    df = data[cols].copy()

    if y_var in df.columns:
        df[y_var] = pd.Categorical(df[y_var], categories=categories[y_var])
    for xvar in x_vars:
        if xvar in df.columns:
            df[xvar] = pd.Categorical(df[xvar], categories=categories[xvar])

    return df


@exareme3_udf(with_aggregation_server=True)
def categorical_nb_fit_udf(
    agg_client,
    data,
    y_var,
    x_vars,
    categories,
):
    df = _prepare_dataframe(data, x_vars, y_var, categories)

    model = CategoricalNB(y_var=y_var, x_vars=x_vars, categories=categories)
    model.fit(df, agg_client)

    category_count = [
        model.category_count[xvar].astype(int).tolist() for xvar in x_vars
    ]
    class_count = model.class_count.astype(int).tolist()
    return {
        "category_count": category_count,
        "class_count": class_count,
    }


@exareme3_udf(with_aggregation_server=True)
def categorical_nb_predict_udf(
    agg_client,
    data,
    y_var,
    x_vars,
    categories,
):
    df = _prepare_dataframe(data, x_vars, y_var, categories)

    model = CategoricalNB(y_var=y_var, x_vars=x_vars, categories=categories)
    model.fit(df, agg_client)

    if df.shape[0] == 0 or model.class_count.sum() == 0:
        return {"predictions": {}}

    preds = model.predict(df[x_vars])
    counts = Counter(preds.tolist())
    predictions = {str(label): int(count) for label, count in counts.items()}
    return {"predictions": predictions}

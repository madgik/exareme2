from collections import Counter
from typing import Dict
from typing import List

from pydantic import BaseModel

from exaflow.algorithms.exareme3.algorithm import Algorithm
from exaflow.algorithms.exareme3.exaflow_registry import exaflow_udf
from exaflow.algorithms.exareme3.naive_bayes_gaussian_model import GaussianNB
from exaflow.worker_communication import BadUserInput

ALGNAME_FIT = "test_nb_gaussian_fit"
ALGNAME_PRED = "test_nb_gaussian_predict"


def _sorted_labels(metadata: dict, y_var: str) -> List[str]:
    return sorted(metadata[y_var]["enumerations"].keys())


def _prepare_dataframe(data, x_vars: List[str], y_var: str):
    cols = list(dict.fromkeys(list(x_vars) + [y_var]))
    return data[cols].copy()


class GaussianNBTestingFit(Algorithm, algname=ALGNAME_FIT):
    class Result(BaseModel):
        theta: List[List[float]]
        var: List[List[float]]
        class_count: List[float]

    def run(self, metadata):
        if not self.inputdata.y or not self.inputdata.x:
            raise BadUserInput("Gaussian NB fit requires X and y.")

        y_var = self.inputdata.y[0]
        x_vars = list(self.inputdata.x)
        labels = _sorted_labels(metadata, y_var)

        udf_results = self.engine.run_algorithm_udf(
            func=gaussian_nb_fit_udf,
            positional_args={
                "inputdata": self.inputdata.json(),
                "y_var": y_var,
                "x_vars": x_vars,
                "labels": labels,
            },
        )

        stats = udf_results[0]
        return self.Result(
            theta=stats["theta"],
            var=stats["var"],
            class_count=stats["class_count"],
        )


class GaussianNBTestingPredict(Algorithm, algname=ALGNAME_PRED):
    class Result(BaseModel):
        predictions: Dict[str, int]

    def run(self, metadata):
        if not self.inputdata.y or not self.inputdata.x:
            raise BadUserInput("Gaussian NB predict requires X and y.")

        y_var = self.inputdata.y[0]
        x_vars = list(self.inputdata.x)
        labels = _sorted_labels(metadata, y_var)

        udf_results = self.engine.run_algorithm_udf(
            func=gaussian_nb_predict_udf,
            positional_args={
                "inputdata": self.inputdata.json(),
                "y_var": y_var,
                "x_vars": x_vars,
                "labels": labels,
            },
        )

        total = Counter()
        for worker_res in udf_results:
            total.update(worker_res["predictions"])

        return self.Result(predictions=dict(total))


@exaflow_udf(with_aggregation_server=True)
def gaussian_nb_fit_udf(
    data,
    inputdata,
    agg_client,
    y_var,
    x_vars,
    labels,
):
    df = _prepare_dataframe(data, x_vars, y_var)

    model = GaussianNB(y_var=y_var, x_vars=x_vars, labels=labels)
    model.fit(df, agg_client)

    theta = model.theta.tolist() if model.theta is not None else []
    var = model.var.tolist() if model.var is not None else []
    class_count = model.class_count.tolist() if model.class_count is not None else []
    return {
        "theta": theta,
        "var": var,
        "class_count": class_count,
    }


@exaflow_udf(with_aggregation_server=True)
def gaussian_nb_predict_udf(
    data,
    inputdata,
    agg_client,
    y_var,
    x_vars,
    labels,
):
    df = _prepare_dataframe(data, x_vars, y_var)

    model = GaussianNB(y_var=y_var, x_vars=x_vars, labels=labels)
    model.fit(df, agg_client)

    if df.shape[0] == 0 or model.total_n_obs == 0 or not model.labels:
        return {"predictions": {}}

    preds = model.predict(df[x_vars])
    counts = Counter(preds.tolist())
    predictions = {str(label): int(count) for label, count in counts.items()}
    return {"predictions": predictions}

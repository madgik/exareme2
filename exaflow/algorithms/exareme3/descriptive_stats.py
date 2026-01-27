from collections import Counter
from functools import reduce
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
from pydantic import BaseModel

from exaflow.algorithms.exareme3.algorithm import Algorithm
from exaflow.algorithms.exareme3.exareme3_registry import exareme3_udf

DATASET_VAR_NAME = "dataset"
ALGORITHM_NAME = "descriptive_stats"


class NumericalDescriptiveStats(BaseModel):
    num_dtps: int
    num_na: int
    num_total: int
    mean: Optional[float]
    std: Optional[float]
    min: Optional[float]
    q1: Optional[float]
    q2: Optional[float]
    q3: Optional[float]
    max: Optional[float]


class NominalDescriptiveStats(BaseModel):
    num_dtps: int
    num_na: int
    num_total: int
    counts: Dict[str, int]


class Variable(BaseModel):
    variable: str
    dataset: str
    data: Union[NominalDescriptiveStats, NumericalDescriptiveStats, None]

    @classmethod
    def from_record(cls, rec):
        data = rec["data"]
        if data is None:
            return cls(**rec)
        if "counts" in data:
            rec["data"] = NominalDescriptiveStats(**data)
        else:
            rec["data"] = NumericalDescriptiveStats(**data)
        return cls(**rec)


class Result(BaseModel):
    variable_based: List[Variable]
    model_based: List[Variable]


class DescriptiveStatisticsAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self, metadata):
        x_vars = self.inputdata.x or []
        y_vars = self.inputdata.y or []
        variable_names = [v for v in (x_vars + y_vars) if v != DATASET_VAR_NAME]

        numerical_vars = [
            var for var in variable_names if not metadata[var]["is_categorical"]
        ]
        nominal_vars = [
            var for var in variable_names if metadata[var]["is_categorical"]
        ]

        local_results = self.engine.run_algorithm_udf(
            func=local_step,
            positional_args={
                "inputdata": self.inputdata.json(),
                "numerical_vars": numerical_vars,
                "nominal_vars": nominal_vars,
                "dropna": False,
                "include_dataset": True,
            },
        )

        recs_varbased = [
            rec for worker_res in local_results for rec in worker_res["recs_varbased"]
        ]
        recs_modbased = [
            rec for worker_res in local_results for rec in worker_res["recs_modbased"]
        ]

        append_missing_datasets(variable_names, self.inputdata.datasets, recs_varbased)
        append_missing_datasets(variable_names, self.inputdata.datasets, recs_modbased)

        recs_varbased += [
            reduce_recs_for_var(recs_varbased, var) for var in variable_names
        ]
        recs_modbased += [
            reduce_recs_for_var(recs_modbased, var) for var in variable_names
        ]

        return Result(
            variable_based=[Variable.from_record(rec) for rec in recs_varbased],
            model_based=[Variable.from_record(rec) for rec in recs_modbased],
        )


@exareme3_udf()
def local_step(data, inputdata, numerical_vars, nominal_vars):
    from exaflow.worker import config as worker_config

    min_row_count = worker_config.privacy.minimum_row_count
    if "dataset" in data.columns:
        ds = data["dataset"]
        if isinstance(ds, pd.DataFrame):
            data["dataset"] = ds.iloc[:, 0]

    return _compute_local_stats(data, numerical_vars, nominal_vars, min_row_count)


def _compute_local_stats(data, numerical_vars, nominal_vars, min_row_count):
    variables = numerical_vars + nominal_vars

    def record(var, dataset, payload):
        return dict(variable=var, dataset=dataset, data=payload)

    def get_empty_records(dataset):
        return [record(var, dataset, None) for var in variables]

    def compute_records(df: pd.DataFrame, dataset: str):
        if len(df) < min_row_count:
            return get_empty_records(dataset)
        num_total = len(df)
        descr_all = df.describe(include="all")
        num_dtps = descr_all.loc["count"].to_dict()
        num_na = {var: num_total - num_dtps.get(var, 0) for var in variables}

        descr_numerical = df.describe()
        if (df.dtypes != "object").any():
            min_ = descr_numerical.loc["min"].to_dict()
            max_ = descr_numerical.loc["max"].to_dict()
            q1 = descr_numerical.loc["25%"].to_dict()
            q2 = descr_numerical.loc["50%"].to_dict()
            q3 = descr_numerical.loc["75%"].to_dict()
            mean = descr_numerical.loc["mean"].to_dict()
            std = descr_numerical.loc["std"].to_dict()
            sx = df[numerical_vars].sum().to_dict() if numerical_vars else {}
            sxx = ((df[numerical_vars] ** 2).sum().to_dict()) if numerical_vars else {}
        counts = {var: df[var].value_counts().to_dict() for var in nominal_vars}

        def numerical_var_data(var):
            if num_dtps.get(var, 0) < min_row_count:
                return None
            return dict(
                num_dtps=num_dtps[var],
                num_na=num_na[var],
                num_total=num_total,
                sx=sx[var],
                sxx=sxx[var],
                q1=q1[var],
                q2=q2[var],
                q3=q3[var],
                mean=mean[var],
                std=None if np.isnan(std[var]) else std[var],
                min=min_[var],
                max=max_[var],
            )

        def nominal_var_data(var):
            if num_dtps.get(var, 0) < min_row_count:
                return None
            return dict(
                num_dtps=num_dtps[var],
                num_na=num_na[var],
                num_total=num_total,
                counts=counts[var],
            )

        numerical_recs = [
            record(var, dataset, numerical_var_data(var)) for var in numerical_vars
        ]
        nominal_recs = [
            record(var, dataset, nominal_var_data(var)) for var in nominal_vars
        ]
        return numerical_recs + nominal_recs

    datasets = set(data["dataset"]) if "dataset" in data.columns else ["unknown"]

    recs_varbased = [
        stats
        for dataset in datasets
        for stats in compute_records(data[data["dataset"] == dataset], dataset)
    ]
    data_nona = data.dropna()
    recs_modbased = [
        stats
        for dataset in datasets
        for stats in compute_records(data_nona[data_nona.dataset == dataset], dataset)
    ]

    return {
        "recs_varbased": recs_varbased,
        "recs_modbased": recs_modbased,
    }


def reduce_recs_for_var(records, variable):
    var_records = [r for r in records if r["variable"] == variable and not isempty(r)]
    if not var_records:
        return dict(variable=variable, dataset="all datasets", data=None)
    reduced_data = reduce(add_records, [rec["data"] for rec in var_records])
    if len(var_records) == 1:
        reduced_data = {
            k: v for k, v in reduced_data.items() if k not in ("q1", "q2", "q3")
        }
    return dict(variable=variable, dataset="all datasets", data=reduced_data)


def add_records(r1, r2):
    result = {}
    for key in ["num_dtps", "num_na", "num_total"]:
        result[key] = r1[key] + r2[key]
    if "counts" in r1:
        result["counts"] = Counter(r1["counts"]) + Counter(r2["counts"])
        return result
    for key in ["sx", "sxx"]:
        result[key] = r1[key] + r2[key]
    result["min"] = min(r1["min"], r2["min"])
    result["max"] = max(r1["max"], r2["max"])
    sx = result["sx"]
    sxx = result["sxx"]
    num = result["num_dtps"]
    mean = sx / num
    result["mean"] = mean
    variance = sxx / (num - 1) - 2 * mean * sx / (num - 1) + num / (num - 1) * mean**2
    result["std"] = None if np.isnan(variance) else float(np.sqrt(max(variance, 0.0)))
    return result


def isempty(rec):
    return rec["data"] is None


def append_missing_datasets(vars, datasets, records):
    def missing(var, dataset):
        return not any(
            r for r in records if r["variable"] == var and r["dataset"] == dataset
        )

    for var in vars:
        for dataset in datasets:
            if missing(var, dataset):
                records.append({"variable": var, "dataset": dataset, "data": None})

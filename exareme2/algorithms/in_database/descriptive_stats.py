"""Descriptive statistics for the MIP

This algorithm differs from the others in some respects. It follows a different
privacy policy and its use-case is very MIP specific.

Algorithm outline
-----------------
For  every  variable  in  x  and  y,  except `dataset`, we compute two groups of
statistics.  The  first  group  is called "variable based" and the second "model
based".

In  the first case every variable is treated independently from the rest. In the
second  group,  the  variables  are  treated as a single model. This means that,
every  variable  being  a  column in the model table, we filter out every row in
that  table  that  contains at least one NA. This means that some variables will
have  values  thrown  out,  if  they  happen  to  be  aligned  with NAs of other
variables.

For  each  group,  variable statistics are computed for each dataset and for all
datasets  globally.  These  statistics  are  returned  as records, each having a
variable name, a dataset name and the statistics data.

The privacy restriction is as follows: if some record has a number of datapoints
less  than  MIN_ROW_COUNT  then  no data is returned for this record. Thus, this
particular variable/dataset pair doesn't contribute to the global computation.
"""
import math
from collections import Counter
from functools import reduce
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy
import pandas as pd
from pydantic import BaseModel

from exareme2.algorithms.in_database.algorithm import Algorithm
from exareme2.algorithms.in_database.algorithm import AlgorithmDataLoader
from exareme2.algorithms.in_database.helpers import get_transfer_data
from exareme2.algorithms.in_database.specifications import AlgorithmName
from exareme2.algorithms.in_database.udfgen import MIN_ROW_COUNT
from exareme2.algorithms.in_database.udfgen import literal
from exareme2.algorithms.in_database.udfgen import merge_transfer
from exareme2.algorithms.in_database.udfgen import relation
from exareme2.algorithms.in_database.udfgen import transfer
from exareme2.algorithms.in_database.udfgen import udf

DATASET_VAR_NAME = "dataset"
ALGORITHM_NAME = AlgorithmName.DESCRIPTIVE_STATS


class DescriptiveStatisticsDataLoader(AlgorithmDataLoader, algname=ALGORITHM_NAME):
    def get_variable_groups(self):
        xvars = self._variables.x
        yvars = self._variables.y

        # dataset variable is special as it is used to group results. Thus, it
        # doesn't make sense to include it also as variable.
        vars = [var for var in xvars + yvars if var != DATASET_VAR_NAME]
        variable_groups = [vars + [DATASET_VAR_NAME]]

        return variable_groups

    def get_dropna(self) -> bool:
        return False

    def get_check_min_rows(self) -> bool:
        return False


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
        if isempty(rec):  # empty records are already formated
            pass
        elif "counts" in rec["data"]:  # record of nominal var
            rec["data"] = NominalDescriptiveStats(**rec["data"])
        else:  # record of numerical var
            rec["data"] = NumericalDescriptiveStats(**rec["data"])
        return cls(**rec)


class Result(BaseModel):
    variable_based: List[Variable]
    model_based: List[Variable]


class DescriptiveStatisticsAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self, data, metadata):
        local_run = self.engine.run_udf_on_local_nodes
        global_run = self.engine.run_udf_on_global_node

        [data] = data
        metadata = metadata
        datasets = self.datasets

        vars = [v for v in data.columns if v != DATASET_VAR_NAME]

        numerical_vars = [var for var in vars if not metadata[var]["is_categorical"]]
        nominal_vars = [var for var in vars if metadata[var]["is_categorical"]]
        local_transfers = local_run(
            func=local,
            keyword_args={
                "data": data,
                "numerical_vars": numerical_vars,
                "nominal_vars": nominal_vars,
            },
            share_to_global=True,
        )

        # global udf is actually a nop but is necessary because we can't get_table_data
        # from local nodes
        local_transfers = get_transfer_data(
            global_run(
                func=global_,
                keyword_args={
                    "local_transfers": local_transfers,
                },
            )
        )

        recs_varbased = [
            rec
            for local_records in local_transfers
            for rec in local_records["recs_varbased"]
        ]
        recs_modbased = [
            rec
            for local_records in local_transfers
            for rec in local_records["recs_modbased"]
        ]

        append_missing_datasets(vars, datasets, recs_varbased)
        append_missing_datasets(vars, datasets, recs_modbased)

        # add global records for each var
        recs_varbased += [reduce_recs_for_var(recs_varbased, var) for var in vars]
        recs_modbased += [reduce_recs_for_var(recs_modbased, var) for var in vars]

        result = Result(
            variable_based=[Variable.from_record(rec) for rec in recs_varbased],
            model_based=[Variable.from_record(rec) for rec in recs_modbased],
        )
        return result


@udf(
    data=relation(),
    numerical_vars=literal(),
    nominal_vars=literal(),
    min_row_count=MIN_ROW_COUNT,
    return_type=transfer(),
)
def local(
    data: pd.DataFrame,
    numerical_vars: list,
    nominal_vars: list,
    min_row_count: int,
):
    vars = numerical_vars + nominal_vars

    def record(var, dataset, data):
        return dict(variable=var, dataset=dataset, data=data)

    def get_empty_records(dataset):
        return [record(var, dataset, None) for var in numerical_vars + nominal_vars]

    def compute_records(data, dataset):
        if len(data) < min_row_count:
            return get_empty_records(dataset)
        num_total = len(data)
        # number datapoints/NA
        descr_all = data.describe(include="all")
        num_dtps = descr_all.loc["count"].to_dict()
        num_na = {var: num_total - num_dtps[var] for var in vars}
        # numerical stats
        descr_numerical = data.describe()
        if (data.dtypes != "object").any():  # check if data has any numerical var
            min_ = descr_numerical.loc["min"].to_dict()
            max_ = descr_numerical.loc["max"].to_dict()
            q1 = descr_numerical.loc["25%"].to_dict()
            q2 = descr_numerical.loc["50%"].to_dict()
            q3 = descr_numerical.loc["75%"].to_dict()
            mean = descr_numerical.loc["mean"].to_dict()
            std = descr_numerical.loc["std"].to_dict()
            sx = data[numerical_vars].sum().to_dict()
            sxx = (data[numerical_vars] ** 2).sum().to_dict()
        # nominal stats
        counts = {var: data[var].value_counts().to_dict() for var in nominal_vars}

        def numerical_var_data(var):
            # if privacy threshold is not met, return empty record
            if num_dtps[var] < min_row_count:
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
                std=None if numpy.isnan(std[var]) else std[var],
                min=min_[var],
                max=max_[var],
            )

        def nominal_var_data(var):
            # if privacy threshold is not met, return empty record
            if num_dtps[var] < min_row_count:
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

    # pandas' groupby skips group if empty, so I implement my own
    def group_by_dataset(data, datasets):
        for dataset in datasets:
            yield dataset, data[data.dataset == dataset]

    # Get dataset enums from actual data. Even if dataset is in variables, we don't
    # need to group by all enums found in the CDEs.
    datasets = list(data.dataset.unique())

    transfer_ = {}
    # compute variable based records
    transfer_["recs_varbased"] = [
        stats
        for dataset, group in group_by_dataset(data, datasets)
        for stats in compute_records(group, dataset)
    ]
    # drop rows with NAs and compute model based records
    data_nona = data.dropna()
    transfer_["recs_modbased"] = [
        stats
        for dataset, group in group_by_dataset(data_nona, datasets)
        for stats in compute_records(group, dataset)
    ]
    return transfer_


@udf(local_transfers=merge_transfer(), return_type=transfer())
def global_(local_transfers):
    return local_transfers


def reduce_recs_for_var(records, variable):
    var_records = [r for r in records if r["variable"] == variable and not isempty(r)]
    if not var_records:
        return dict(variable=variable, dataset="all datasets", data=None)
    reduced_data = reduce(add_records, [rec["data"] for rec in var_records])
    # If  len(var_records)  >  1  then  add_records is called, which doesn't
    # compute  quartiles  globally.  However,  when  len(var_records)  == 1,
    # reduce  returns  the first element. In that case remove quartiles from
    # global result.
    if len(var_records) == 1:
        reduced_data = {
            k: v for k, v in reduced_data.items() if k not in ("q1", "q2", "q3")
        }
    return dict(
        variable=variable,
        dataset="all datasets",
        data=reduced_data,
    )


def add_records(r1, r2):
    # common fields
    result = {}
    for key in ["num_dtps", "num_na", "num_total"]:
        result[key] = r1[key] + r2[key]
    # nominal fields
    if "counts" in r1:
        result["counts"] = Counter(r1["counts"]) + Counter(r2["counts"])
        return result
    # numerical fields
    for key in ["sx", "sxx"]:
        result[key] = r1[key] + r2[key]
    result["min"] = min(r1["min"], r2["min"])
    result["max"] = max(r1["max"], r2["max"])
    # compute global mean and std
    sx = result["sx"]
    sxx = result["sxx"]
    num = result["num_dtps"]
    mean = sx / num
    result["mean"] = mean
    variance = sxx / (num - 1) - 2 * mean * sx / (num - 1) + num / (num - 1) * mean**2
    result["std"] = None if numpy.isnan(variance) else math.sqrt(variance)
    return result


def isempty(rec):
    return rec["data"] is None


def append_missing_datasets(vars, datasets, records):
    # When a user selects some datasets, they should all apear in the result.
    # Some datasets might have empty records, however. This function appends
    # the missing datasets to the results. For each missing datasets we need
    # to add records for each variable.
    def all_different(var, dataset):
        return not any(
            record
            for record in records
            if record["variable"] == var and record["dataset"] == dataset
        )

    combos = ((var, dataset) for var in vars for dataset in datasets)

    for var, dataset in combos:
        if all_different(var, dataset):
            records.append({"variable": var, "dataset": dataset, "data": None})

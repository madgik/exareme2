from typing import List
from typing import Optional
from typing import TypeVar
from typing import Union

import numpy
from pydantic import BaseModel

from mipengine.algorithms.base_classes.algorithm import Algorithm
from mipengine.algorithms.helpers import get_transfer_data
from mipengine.algorithms.specifications.algorithm_specification import AlgorithmName
from mipengine.algorithms.specifications.algorithm_specification import (
    AlgorithmSpecification,
)
from mipengine.algorithms.specifications.inputdata_specification import (
    InputDataSpecification,
)
from mipengine.algorithms.specifications.inputdata_specification import (
    InputDataSpecifications,
)
from mipengine.algorithms.specifications.inputdata_specification import (
    InputDataStatType,
)
from mipengine.algorithms.specifications.inputdata_specification import InputDataType
from mipengine.algorithms.specifications.parameter_specification import (
    ParameterSpecification,
)
from mipengine.algorithms.specifications.parameter_specification import ParameterType
from mipengine.udfgen import MIN_ROW_COUNT
from mipengine.udfgen import literal
from mipengine.udfgen import merge_transfer
from mipengine.udfgen import relation
from mipengine.udfgen import secure_transfer
from mipengine.udfgen import transfer
from mipengine.udfgen import udf

S = TypeVar("S")


class Histogram(BaseModel):
    var: str  # y
    grouping_var: Optional[str]  # x[i]
    grouping_enum: Optional[str]  # enum of x[i]
    bins: List[Union[float, str]]
    counts: List[Optional[int]]


class HistogramResult1(BaseModel):
    histogram: List[Histogram]


class MultipleHistogramsAlgorithm(
    Algorithm, stepname=AlgorithmName.MULTIPLE_HISTOGRAMS.value
):
    @classmethod
    def get_specification(cls):
        return AlgorithmSpecification(
            name=cls.stepname,
            desc="Multiple Histograms",
            label="Multiple Histograms",
            enabled=True,
            inputdata=InputDataSpecifications(
                y=InputDataSpecification(
                    label="y",
                    desc="Variable to distribute among bins.",
                    types=[InputDataType.REAL, InputDataType.INT, InputDataType.TEXT],
                    stattypes=[InputDataStatType.NUMERICAL, InputDataStatType.NOMINAL],
                    notblank=True,
                    multiple=False,
                ),
                x=InputDataSpecification(
                    label="x",
                    desc="Nominal variable for grouping bins.",
                    types=[InputDataType.INT, InputDataType.TEXT],
                    stattypes=[InputDataStatType.NOMINAL],
                    notblank=False,
                    multiple=True,
                ),
            ),
            parameters={
                "bins": ParameterSpecification(
                    label="Number of bins",
                    desc="Number of bins",
                    types=[ParameterType.INT],
                    notblank=False,
                    multiple=False,
                    default=20,
                    min=1,
                    max=100,
                ),
            },
        )

    def get_variable_groups(self):
        return [self.variables.y + self.variables.x]

    def run(self, engine):
        local_run = engine.run_udf_on_local_nodes
        global_run = engine.run_udf_on_global_node

        xvars = self.variables.x
        yvars = self.variables.y
        yvar = yvars[0]

        default_bins = 20
        bins = self.algorithm_parameters.get("bins", default_bins)
        if bins is None:
            bins = default_bins

        [data] = engine.data_model_views

        metadata = dict(self.metadata)

        vars = [var for var in xvars + yvars]

        nominal_vars = [var for var in vars if metadata[var]["is_categorical"]]

        enumerations_dict = {var: metadata[var]["enumerations"] for var in nominal_vars}

        if yvar in xvars:
            xvars.remove(yvar)

        if yvar in nominal_vars:
            locals_result = local_run(
                func=compute_local_histogram_categorical_secure,
                positional_args=[data, enumerations_dict, yvar, xvars],
                share_to_global=[True],
            )

            categorical_histogram = get_transfer_data(
                global_run(
                    func=merge_grouped_histogram_categorical_secure2,
                    positional_args=[locals_result, xvars],
                    share_to_locals=[False],
                )
            )
            numerical_histogram = {}

        else:
            local_result4 = local_run(
                func=find_min_max_local,
                positional_args=[data, yvar],
                share_to_global=[True],
            )
            min_max_result = get_transfer_data(
                global_run(
                    func=find_min_max_global,
                    positional_args=[local_result4],
                    share_to_locals=[False],
                )
            )
            min_value = min_max_result["min_value"]
            max_value = min_max_result["max_value"]
            local_result5, bins_transfer = local_run(
                func=compute_local_histogram_numerical_secure2,
                positional_args=[
                    data,
                    enumerations_dict,
                    bins,
                    yvar,
                    xvars,
                    min_value,
                    max_value,
                ],
                share_to_global=[True, True],
            )

            numerical_histogram = get_transfer_data(
                global_run(
                    func=merge_grouped_histogram_numerical_secure2,
                    positional_args=[local_result5, bins_transfer, xvars],
                    share_to_locals=[False],
                )
            )
            categorical_histogram = {}

        return_list = []
        if yvar in nominal_vars:
            categorical_histogram1 = Histogram(
                var=yvar,
                bins=list(enumerations_dict[yvar].keys()),
                counts=categorical_histogram["categorical_histogram"],
            )
            return_list.append(categorical_histogram1)
            if xvars:
                for i, x_variable in enumerate(xvars):
                    possible_groups = enumerations_dict[x_variable].keys()
                    possible_values = list(enumerations_dict[yvar].keys())
                    for j, curr_group in enumerate(possible_groups):
                        curr_group_histogram = Histogram(
                            var=yvar,
                            grouping_var=x_variable,
                            grouping_enum=curr_group,
                            bins=possible_values,
                            counts=categorical_histogram[
                                "grouped_histogram_categorical"
                            ][i][j],
                        )
                        return_list.append(curr_group_histogram)

        else:
            numerical_histogram1 = Histogram(
                var=yvar,
                bins=numerical_histogram["numerical_bins"],
                counts=numerical_histogram["histogram"],
            )
            return_list.append(numerical_histogram1)
            if xvars:
                for i, x_variable in enumerate(xvars):
                    possible_groups = enumerations_dict[x_variable].keys()
                    for j, curr_group in enumerate(possible_groups):
                        curr_group_histogram = Histogram(
                            var=yvar,
                            grouping_var=x_variable,
                            grouping_enum=curr_group,
                            bins=numerical_histogram["numerical_bins"],
                            counts=numerical_histogram["grouped_histogram"][i][j],
                        )
                        return_list.append(curr_group_histogram)

        ret_val = HistogramResult1(histogram=return_list)
        return ret_val


@udf(
    data=relation(S),
    column=literal(),
    return_type=[secure_transfer(sum_op=True, min_op=True, max_op=True)],
)
def find_min_max_local(data, column):
    min_value = data[column].min()
    max_value = data[column].max()

    secure_transfer_ = {
        "min": {"data": float(min_value), "operation": "min", "type": "float"},
        "max": {"data": float(max_value), "operation": "max", "type": "float"},
    }

    return secure_transfer_


@udf(
    locals_result=secure_transfer(sum_op=True, min_op=True, max_op=True),
    return_type=[transfer()],
)
def find_min_max_global(locals_result):

    transfer_ = {
        "min_value": locals_result["min"],
        "max_value": locals_result["max"],
    }

    return transfer_


@udf(
    data=relation(S),
    metadata=literal(),
    yvar=literal(),
    xvars=literal(),
    return_type=[secure_transfer(sum_op=True, min_op=True, max_op=True)],
)
def compute_local_histogram_categorical_secure(data, metadata, yvar, xvars):
    possible_enumerations = metadata[yvar].keys()
    local_counts = data[yvar].value_counts().to_dict()
    categorical_histogram_list = []
    for curr_key in possible_enumerations:
        curr_value = local_counts.get(curr_key, 0)
        categorical_histogram_list.append(curr_value)

    secure_transfer_ = {}
    secure_transfer_["categorical_histogram"] = {
        "data": categorical_histogram_list,
        "operation": "sum",
        "type": "int",
    }
    if xvars:
        final_dict = {}
        for x_variable in xvars:
            local_grouped_histogram = {}
            grouped = data[[yvar, x_variable]].groupby(x_variable)
            for group_name, curr_grouped in grouped:
                local_grouped_histogram[group_name] = (
                    curr_grouped[yvar].value_counts().to_dict()
                )
            final_dict[x_variable] = local_grouped_histogram

            possible_groups = metadata[x_variable].keys()

            for curr_group in possible_groups:
                curr_result = final_dict[x_variable].get(curr_group, {})
                final_dict[x_variable][curr_group] = curr_result

        grouped_list = []
        for x_variable in xvars:
            possible_groups = metadata[x_variable].keys()
            possible_values = metadata[yvar].keys()
            groups_list = []
            for curr_group in possible_groups:
                elements_list = []
                for curr_element in possible_values:
                    curr_result = final_dict[x_variable][curr_group].get(
                        curr_element, 0
                    )
                    elements_list.append(curr_result)
                groups_list.append(elements_list)
            grouped_list.append(groups_list)
        secure_transfer_["grouped_histogram_categorical"] = {
            "data": grouped_list,
            "operation": "sum",
            "type": "int",
        }
    return secure_transfer_


@udf(
    locals_result=secure_transfer(sum_op=True, min_op=True, max_op=True),
    xvars=literal(),
    min_row_count=MIN_ROW_COUNT,
    return_type=[transfer()],
)
def merge_grouped_histogram_categorical_secure2(locals_result, xvars, min_row_count):
    return_dict = {}
    histogram_merge = locals_result["categorical_histogram"]
    return_dict["categorical_histogram"] = [
        curr_value if curr_value >= min_row_count else None
        for curr_value in histogram_merge
    ]
    if xvars:
        grouped_merge = locals_result["grouped_histogram_categorical"]
        x_variable_return = []
        for x_variable in grouped_merge:
            group_list = []
            for curr_group in x_variable:
                elements_list = []
                for curr_element in curr_group:
                    curr_value = curr_element if curr_element >= min_row_count else None
                    elements_list.append(curr_value)
                group_list.append(elements_list)
            x_variable_return.append(group_list)
        return_dict["grouped_histogram_categorical"] = x_variable_return
    return return_dict


@udf(
    data=relation(S),
    metadata=literal(),
    bins=literal(),
    yvar=literal(),
    xvars=literal(),
    min_value=literal(),
    max_value=literal(),
    return_type=[secure_transfer(sum_op=True, min_op=True, max_op=True), transfer()],
)
def compute_local_histogram_numerical_secure2(
    data, metadata, bins, yvar, xvars, min_value, max_value
):
    def hist_func(x, min_value, max_value, bins=20):
        hist, bins = numpy.histogram(x, range=(min_value, max_value), bins=bins)
        return hist

    final_dict = {}
    local_histogram, local_bins = numpy.histogram(
        data[yvar], range=(min_value, max_value), bins=bins
    )

    secure_transfer_ = {}
    secure_transfer_["histogram"] = {
        "data": local_histogram.tolist(),
        "operation": "sum",
        "type": "int",
    }
    transfer_ = {}
    transfer_["bins"] = local_bins.tolist()
    if xvars:
        final_dict = {}
        for x_variable in xvars:
            local_grouped_histogram = (
                data[[yvar, x_variable]]
                .groupby(x_variable)
                .apply(
                    lambda x: hist_func(
                        x, min_value=min_value, max_value=max_value, bins=bins
                    ).tolist()
                )
                .to_dict()
            )
            final_dict[x_variable] = local_grouped_histogram

        for x_variable in xvars:
            for curr_group in metadata[x_variable].keys():
                result = final_dict[x_variable].get(
                    curr_group, numpy.zeros(bins, dtype="int64").tolist()
                )
                final_dict[x_variable][curr_group] = result
        x_variables_list = []
        for x_variable in xvars:
            groups_list = []
            for curr_group in metadata[x_variable].keys():
                curr_element = final_dict[x_variable][curr_group]
                groups_list.append(curr_element)
            x_variables_list.append(groups_list)

        secure_transfer_["grouped_histogram"] = {
            "data": x_variables_list,
            "operation": "sum",
            "type": "int",
        }

    return secure_transfer_, transfer_


@udf(
    locals_result=secure_transfer(sum_op=True, min_op=True, max_op=True),
    bins_transfer=merge_transfer(),
    xvars=literal(),
    min_row_count=MIN_ROW_COUNT,
    return_type=[transfer()],
)
def merge_grouped_histogram_numerical_secure2(
    locals_result, bins_transfer, xvars, min_row_count
):
    return_dict = {}
    histogram_merge = locals_result["histogram"]
    return_dict["histogram"] = [
        curr_value if curr_value >= min_row_count else None
        for curr_value in histogram_merge
    ]
    if xvars:
        grouped_merge = locals_result["grouped_histogram"]
        x_variable_return = []
        for x_variable in grouped_merge:
            group_list = []
            for curr_group in x_variable:
                elements_list = []
                for curr_element in curr_group:
                    curr_value = curr_element if curr_element >= min_row_count else None
                    elements_list.append(curr_value)
                group_list.append(elements_list)
            x_variable_return.append(group_list)
        return_dict["grouped_histogram"] = x_variable_return
    return_dict["numerical_bins"] = bins_transfer[0]["bins"]
    return return_dict

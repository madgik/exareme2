import numpy
import pandas as pd

from tests.testcase_generators.testcase_generator import TestCaseGenerator


def compute_histogram_categorical(data, enums, yvar, xvars):
    possible_enumerations = enums[yvar].keys()
    local_counts = data[yvar].value_counts().to_dict()
    categorical_histogram_list = []
    grouped_list = None
    for curr_key in sorted(possible_enumerations):
        curr_value = local_counts.get(curr_key, 0)
        categorical_histogram_list.append(curr_value)

    if xvars:
        final_dict = {}
        for x_variable in xvars:
            # print(data[[yvar, x_variable]])
            local_grouped_histogram = {}
            grouped = data[[yvar, x_variable]].groupby(x_variable)
            for group_name, curr_grouped in grouped:
                local_grouped_histogram[group_name] = (
                    curr_grouped[yvar].value_counts().to_dict()
                )
                # print(local_grouped_histogram[group_name])
            final_dict[x_variable] = local_grouped_histogram

            possible_groups = sorted(enums[x_variable].keys())

            for curr_group in possible_groups:
                curr_result = final_dict[x_variable].get(curr_group, {})
                final_dict[x_variable][curr_group] = curr_result

        grouped_list = []
        # print(final_dict)
        for x_variable in xvars:
            possible_groups = sorted(enums[x_variable].keys())
            possible_values = sorted(enums[yvar].keys())
            groups_list = []
            for curr_group in possible_groups:
                elements_list = []
                for curr_element in possible_values:
                    curr_result = final_dict[x_variable][curr_group].get(
                        curr_element, 0
                    )
                    elements_list.append(curr_result)
                print(elements_list)
                groups_list.append(elements_list)
            grouped_list.append(groups_list)

    return categorical_histogram_list, grouped_list


def compute_numerical_histogram(data, enums, bins, yvar, xvars, min_value, max_value):
    x_variables_list = None
    # yvar = algo_interface.y_variables[0]
    def hist_func(x, min_value, max_value, bins=20):
        hist, bins = numpy.histogram(x, range=(min_value, max_value), bins=bins)
        return hist

    final_dict = {}
    local_histogram = hist_func(
        data[yvar], min_value=min_value, max_value=max_value, bins=bins
    )

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
        # transfer_={}
        # transfer_['grouped_histogram']= final_dict

        for x_variable in xvars:
            for curr_group in sorted(enums[x_variable].keys()):
                result = final_dict[x_variable].get(
                    curr_group, numpy.zeros(bins, dtype="int64").tolist()
                )
                final_dict[x_variable][curr_group] = result
        x_variables_list = []
        for x_variable in xvars:
            groups_list = []
            for curr_group in sorted(enums[x_variable].keys()):
                curr_element = final_dict[x_variable][curr_group]
                groups_list.append(curr_element)
            x_variables_list.append(groups_list)

    return local_histogram.tolist(), x_variables_list


class HistogramTestcaseGenerator(TestCaseGenerator):
    full_data = False
    dropna = True

    def compute_expected_output(self, input_data, parameters, metadata: dict):

        numerical_vars = [md["code"] for md in metadata if not md["isCategorical"]]
        nominal_vars = [
            md["code"]
            for md in metadata
            if md["isCategorical"] and md["code"] != "dataset"
        ]
        vars = numerical_vars + nominal_vars
        enums = {
            var: next(md["enumerations"] for md in metadata if md["code"] == var)
            for var in nominal_vars
        }

        # print(enums)
        enums2 = {}
        for key, value in enums.items():
            enums2[key] = {}
            for curr_element in value:
                enums2[key][curr_element["code"]] = curr_element["label"]
        # print(enums2)
        enums = enums2
        bins = parameters["bins"]
        return_dict = {}
        Y, X = input_data
        Y_data = pd.DataFrame(Y)
        X_data = pd.DataFrame(X)
        if not Y_data.empty:
            yvars = list(Y.columns.values)
        else:
            yvars = []
        yvar = yvars[0]
        if not X_data.empty:
            xvars = list(X.columns.values)
        else:
            xvars = []
        data = pd.concat([Y_data, X_data])
        data = data.dropna()
        if data.empty:
            return None
        all_vars = yvars + xvars
        if "dataset" in all_vars:
            return None
        if yvar in nominal_vars:
            # categorical histogram
            categorical_histogram_list, grouped_list = compute_histogram_categorical(
                data, enums, yvar, xvars
            )
            return_dict["numerical"] = {}
            return_dict["categorical"] = {}
            return_dict["categorical"][
                "categorical_histogram"
            ] = categorical_histogram_list
            if xvars:
                return_dict["categorical"][
                    "grouped_histogram_categorical"
                ] = grouped_list
        else:
            # numerical histogram
            min_value = data[yvar].min()
            max_value = data[yvar].max()

            local_histogram, x_variables_list = compute_numerical_histogram(
                data, enums, bins, yvar, xvars, min_value, max_value
            )

            return_dict["categorical"] = {}
            return_dict["numerical"] = {}
            return_dict["numerical"]["histogram"] = local_histogram
            if xvars:
                return_dict["numerical"]["grouped_histogram"] = x_variables_list
        return return_dict


if __name__ == "__main__":
    with open("mipengine/algorithms/histogram.json") as specs_file:
        gen = HistogramTestcaseGenerator(specs_file)
    with open("histogram_tmp.json", "w") as expected_file:
        gen.write_test_cases(expected_file, 100)

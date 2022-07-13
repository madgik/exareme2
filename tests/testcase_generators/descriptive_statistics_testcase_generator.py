import numpy
import pandas as pd

from tests.testcase_generators.testcase_generator import TestCaseGenerator

class DesciptiveStatisticsTestCaseGenerator(TestCaseGenerator):
    def compute_expected_output(self, input_data, input_parameters=None,datatypes=None):
        X_dataset, Y_dataset = input_data

        full_dataset = pd.concat([X_dataset,Y_dataset],axis=1)
        all_columns_dataset = list(full_dataset.columns.values)

        numerical_columns = datatypes['numerical']
        categorical_columns = datatypes['nominal']

        numerical_columns_dataset = sorted(list(set(all_columns_dataset).intersection(set(numerical_columns))))
        categorical_columns_dataset = sorted(list(set(all_columns_dataset).intersection(set(categorical_columns))))

        X = full_dataset[numerical_columns_dataset].values
        categoricals_df = full_dataset[categorical_columns_dataset]

        categorical_counts = []

        for curr_categorical_column in categorical_columns_dataset:
            filled_values1 = categoricals_df[curr_categorical_column].fillna("NaN")
            categorical_counts.append(filled_values1.value_counts(dropna=False).to_dict())

        X_mean = numpy.nanmean(X,axis=0)
        X_max = numpy.nanmax(X,axis=0)
        X_min = numpy.nanmin(X,axis=0)
        X_std = numpy.nanstd(X,axis=0,ddof=1)

        X_not_null = X[~numpy.isnan(X).any(axis=1)]

        model_mean = numpy.nanmean(X_not_null,axis=0)
        model_max = numpy.nanmax(X_not_null,axis=0)
        model_min = numpy.nanmin(X_not_null,axis=0)
        model_std = numpy.nanstd(X_not_null,axis=0,ddof=1)


        output = {
            "max": X_max.tolist(),
            "min": X_min.tolist(),
            "mean": X_mean.tolist(),
            'std': X_std.tolist(),
            'max_model' : model_max.tolist(),
            'min_model' : model_min.tolist(),
            "mean_model": model_mean.tolist(),
            'std_model' : model_std.tolist(),
            'categorical_counts': categorical_counts,
            'numerical_columns' : numerical_columns_dataset,
            'categorical_columns' : categorical_columns_dataset

        }
        return output


if __name__ == "__main__":
    with open("mipengine/algorithms/descriptive_statistics.json") as specs_file:
        descgen = DesciptiveStatisticsTestCaseGenerator(specs_file)
    with open("tmp.json", "w") as expected_file:
        descgen.write_test_cases(expected_file)

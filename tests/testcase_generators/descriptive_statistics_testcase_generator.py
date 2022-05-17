from sklearn.decomposition import PCA
import numpy

from tests.testcase_generators.testcase_generator import TestCaseGenerator


class DesciptiveStatisticsTestCaseGenerator(TestCaseGenerator):
    def compute_expected_output(self, input_data, input_parameters=None):
        X, _ = input_data

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
            'std_model' : model_std.tolist()

        }
        return output


if __name__ == "__main__":
    with open("mipengine/algorithms/descriptive_statistics.json") as specs_file:
        pcagen = DesciptiveStatisticsTestCaseGenerator(specs_file)
    with open("tmp.json", "w") as expected_file:
        pcagen.write_test_cases(expected_file)

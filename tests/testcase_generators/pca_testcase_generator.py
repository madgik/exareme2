from pathlib import Path

from sklearn.decomposition import PCA

from tests.testcase_generators.testcase_generator import TestCaseGenerator

SPECS_PATH = Path("mipengine", "algorithms", "ttest_paired.json")
EXPECTED_PATH = Path(
    "tests",
    "algorithm_validation_tests",
    "expected",
    "pca_expected.json",
)


class PCATestCaseGenerator(TestCaseGenerator):
    def compute_expected_output(self, input_data, input_parameters=None):
        X, _ = input_data
        X -= X.mean(axis=0)
        X /= X.std(axis=0, ddof=1)
        pca = PCA()
        pca.fit(X)

        output = {
            "n_obs": len(X),
            "eigen_vals": pca.explained_variance_.tolist(),
            "eigen_vecs": pca.components_.tolist(),
        }
        return output


if __name__ == "__main__":
    with open("mipengine/algorithms/pca.json") as specs_file:
        pcagen = PCATestCaseGenerator(specs_file)
    with open(EXPECTED_PATH, "w") as expected_file:
        pcagen.write_test_cases(expected_file)

from sklearn.decomposition import PCA

from testcase_generator import TestCaseGenerator


class PCATestCaseGenerator(TestCaseGenerator):
    def compute_expected_output(self, input_data, input_parameters=None):
        X = input_data
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
    pcagen = PCATestCaseGenerator(
        expected_path="tests/algorithms/expected/tmp.json",
        dataset_path="tests/demo_data/dementia/desd-synthdata.csv",
        variable_types="numerical",
    )
    pcagen.write_test_cases()

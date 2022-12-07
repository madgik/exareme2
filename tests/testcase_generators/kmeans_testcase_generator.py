import json
from typing import List
from typing import Optional
from typing import TypeVar
from typing import Union

import numpy
import pandas as pd
from pydantic import BaseModel
from sklearn.cluster import KMeans

from tests.testcase_generators.testcase_generator import TestCaseGenerator


class KmeansTestcaseGenerator(TestCaseGenerator):
    full_data = False
    dropna = True

    def compute_expected_output(self, input_data, parameters, metadata: dict):

        k = parameters["k"]
        tol = parameters["tol"]
        maxiter = parameters["maxiter"]

        X, Y = input_data

        X_val = X.values

        print(X_val.shape)

        max_vals = numpy.nanmax(X_val, axis=0)
        min_vals = numpy.nanmin(X_val, axis=0)

        random_state = numpy.random.RandomState(seed=123)

        centers_init = random_state.uniform(
            low=min_vals, high=max_vals, size=(k, min_vals.shape[0])
        )

        kmeans = KMeans(n_clusters=k, init=centers_init, max_iter=maxiter, tol=tol)
        kmeans.fit(X_val)

        ret_val = {}
        ret_val["centers"] = kmeans.cluster_centers_.tolist()
        return ret_val


if __name__ == "__main__":
    with open("mipengine/algorithms/kmeans.json") as specs_file:
        gen = KmeansTestcaseGenerator(specs_file)
    with open("kmeans_tmp.json", "w") as expected_file:
        gen.write_test_cases(expected_file, 1)

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
        if X_val.shape[0] == 0:
            print("should exit")
            return None
        if X_val.shape[0] < k:
            print("should exit")
            return None

        max_vals = numpy.nanmax(X_val, axis=0)
        min_vals = numpy.nanmin(X_val, axis=0)

        random_state = numpy.random.RandomState(seed=123)

        centers_init = random_state.uniform(
            low=min_vals, high=max_vals, size=(k, min_vals.shape[0])
        )
        if centers_init.shape[0] == 0:
            return None
        kmeans = KMeans(n_clusters=k, init=centers_init, max_iter=maxiter, tol=tol)
        kmeans.fit(X_val)
        print(k)
        print(kmeans.cluster_centers_.shape)
        ret_val = {}
        ret_val["init_centers"] = centers_init.tolist()
        ret_val["centers"] = kmeans.cluster_centers_.tolist()
        ret_val["n_iter"] = kmeans.n_iter_
        return ret_val


if __name__ == "__main__":
    with open("mipengine/algorithms/kmeans.json") as specs_file:
        gen = KmeansTestcaseGenerator(specs_file)
    with open("kmeans_tmp100.json", "w") as expected_file:
        gen.write_test_cases(expected_file, 100)

import json
from typing import List
from typing import Optional
from typing import TypeVar
from typing import Union

import numpy
import pandas as pd
from pydantic import BaseModel
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

from exareme2.algorithms.random_forest_cv import RandomForestCVAlgorithm
from tests.testcase_generators.testcase_generator import TestCaseGenerator


class RandomForestCVResult(BaseModel):
    title: str  # y
    accuracy_list: List[float]


class RandomForestTestcaseGenerator(TestCaseGenerator):
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

        Y, X = input_data
        Y_data = pd.DataFrame(Y)
        X_data = pd.DataFrame(X)

        le = preprocessing.LabelEncoder()
        Y_conv = Y_data.values.ravel()
        le.fit(Y_conv)
        ydata_enc = le.transform(Y_conv)

        n_splits = parameters["n_splits"]

        num_trees = parameters["n_estimators"]

        kf = KFold(n_splits=n_splits)

        curr_accuracy_list = []
        for train, test in kf.split(X):

            X_train = X.values[train]
            y_train = ydata_enc[train]

            X_test = X.values[test]
            y_test = ydata_enc[test]

            rf = RandomForestClassifier(n_estimators=num_trees)
            rf.fit(X_train, y_train)

            y_pred = rf.predict(X_test)

            from sklearn.metrics import accuracy_score

            curr_accuracy = accuracy_score(y_test, y_pred)
            curr_accuracy_list.append(curr_accuracy)

        ret_val = RandomForestCVResult(
            title="random_forest", accuracy_list=curr_accuracy_list
        )

        return json.loads(ret_val.json())


if __name__ == "__main__":
    specs = json.loads(
        RandomForestCVAlgorithm.get_specification().json(exclude_none=True)
    )
    gen = RandomForestTestcaseGenerator(specs)
    with open("tmp.json", "w") as expected_file:
        gen.write_test_cases(expected_file, num_test_cases=5)

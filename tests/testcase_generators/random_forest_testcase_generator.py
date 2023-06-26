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
        le.fit(Y_data)
        ydata_enc = le.transform(Y_data)

        n_splits = 2

        num_trees = 64

        kf = KFold(n_splits=n_splits)

        curr_accuracy_list = []
        for train, test in kf.split(X):

            X_train = X[train]
            y_train = ydata_enc[train]

            X_test = X[test]
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
    with open("mipengine/algorithms/random_forest_cv.json") as specs_file:
        gen = RandomForestTestcaseGenerator(specs_file)
    with open("new_rf.json", "w") as expected_file:
        gen.write_test_cases(expected_file, 2)

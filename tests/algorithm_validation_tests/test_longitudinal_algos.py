from copy import deepcopy

from tests.algorithm_validation_tests.helpers import algorithm_request

base_input = {
    "inputdata": {
        "y": None,
        "x": None,
        "data_model": "longitudinal_dementia:0.1",
        "datasets": ["longitudinal_dementia"],
        "filters": None,
    },
    "parameters": {
        "visit1": "BL",
        "visit2": "FL1",
        "strategies": None,
    },
    "flags": {"longitudinal": True},
}


def test_longitudinal_anova_oneway():
    input = deepcopy(base_input)
    input["inputdata"]["x"] = ["gender"]
    input["inputdata"]["y"] = ["lefthippocampus"]
    input["parameters"]["strategies"] = {"gender": "first", "lefthippocampus": "diff"}
    response = algorithm_request("anova_oneway", input)
    assert response.status_code == 200


def test_longitudinal_anova_twoway():
    input = deepcopy(base_input)
    input["inputdata"]["x"] = ["gender", "agegroup"]
    input["inputdata"]["y"] = ["lefthippocampus"]
    input["parameters"]["sstype"] = 2
    input["parameters"]["strategies"] = {
        "gender": "first",
        "lefthippocampus": "diff",
        "agegroup": "first",
    }
    response = algorithm_request("anova", input)
    assert response.status_code == 200


def test_longitudinal_linear_regression():
    input = deepcopy(base_input)
    input["inputdata"]["x"] = ["lefthippocampus"]
    input["inputdata"]["y"] = ["righthippocampus"]
    input["parameters"]["strategies"] = {
        "righthippocampus": "first",
        "lefthippocampus": "diff",
    }
    response = algorithm_request("linear_regression", input)
    assert response.status_code == 200


def test_longitudinal_linear_regression_cv():
    input = deepcopy(base_input)
    input["inputdata"]["x"] = ["lefthippocampus"]
    input["inputdata"]["y"] = ["righthippocampus"]
    input["parameters"]["n_splits"] = 2
    input["parameters"]["strategies"] = {
        "righthippocampus": "first",
        "lefthippocampus": "diff",
    }
    response = algorithm_request("linear_regression_cv", input)
    assert response.status_code == 200


def test_longitudinal_logistic_regression():
    input = deepcopy(base_input)
    input["inputdata"]["x"] = ["lefthippocampus", "leftamygdala"]
    input["inputdata"]["y"] = ["gender"]
    input["parameters"]["positive_class"] = "F"
    input["parameters"]["strategies"] = {
        "gender": "second",
        "leftamygdala": "first",
        "lefthippocampus": "diff",
    }
    response = algorithm_request("logistic_regression", input)
    assert response.status_code == 200


def test_longitudinal_logistic_regression_cv():
    input = deepcopy(base_input)
    input["inputdata"]["x"] = ["lefthippocampus", "leftamygdala"]
    input["inputdata"]["y"] = ["gender"]
    input["parameters"]["n_splits"] = 2
    input["parameters"]["positive_class"] = "F"
    input["parameters"]["strategies"] = {
        "gender": "second",
        "leftamygdala": "first",
        "lefthippocampus": "diff",
    }
    response = algorithm_request("logistic_regression_cv", input)
    assert response.status_code == 200


def test_longitudinal_naive_bayes_gaussian_cv():
    input = deepcopy(base_input)
    input["inputdata"]["x"] = ["lefthippocampus", "leftamygdala"]
    input["inputdata"]["y"] = ["gender"]
    input["parameters"]["n_splits"] = 2
    input["parameters"]["strategies"] = {
        "gender": "second",
        "leftamygdala": "first",
        "lefthippocampus": "diff",
    }
    response = algorithm_request("naive_bayes_gaussian_cv", input)
    assert response.status_code == 200


def test_longitudinal_naive_bayes_categorical_cv():
    input = deepcopy(base_input)
    input["inputdata"]["x"] = ["agegroup"]
    input["inputdata"]["y"] = ["gender"]
    input["parameters"]["n_splits"] = 2
    input["parameters"]["strategies"] = {
        "gender": "second",
        "agegroup": "first",
    }
    response = algorithm_request("naive_bayes_categorical_cv", input)
    assert response.status_code == 200

def test_logistic_regression(get_algorithm_result):
    input = {
        "inputdata": {
            "y": ["gender"],
            "x": ["lefthippocampus"],
            "data_model": "dementia:0.1",
            "datasets": [
                "ppmi0",
                "ppmi1",
                "ppmi2",
                "ppmi3",
                "ppmi5",
                "ppmi6",
                "edsd6",
                "ppmi7",
                "ppmi8",
                "ppmi9",
            ],
            "filters": None,
        },
        "parameters": None,
        "test_case_num": 99,
    }
    input["type"] = "flower"
    algorithm_result = get_algorithm_result("logistic_regression", input)
    assert {"accuracy": 0.3813682678311499} == algorithm_result or algorithm_result == {
        "accuracy": 0.61863173216885
    }

def test_logistic_regression_with_filters(get_algorithm_result):
    input = {
        "inputdata": {
            "y": ["rightputamen", "rightsfgsuperiorfrontalgyrus"],
            "data_model": "dementia:0.1",
            "datasets": ["ppmi1", "ppmi2", "ppmi3", "edsd0"],
            "filters": None,
        },
        "parameters": {},
        "test_case_num": 99,
    }
    input["type"] = "exareme2"
    algorithm_result = get_algorithm_result("pca", input)
    print(algorithm_result)

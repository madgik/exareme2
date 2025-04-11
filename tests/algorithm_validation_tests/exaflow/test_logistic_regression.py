def test_logistic_regression_with_filters(get_algorithm_result):
    input = {
        "inputdata": {
            "datasets": ["edsd0", "edsd1"],
            "y": [
                "lefthippocampus",
            ],
            "data_model": "dementia:0.1",
            "validation_datasets": [],
            "filters": {
                "condition": "AND",
                "rules": [
                    {
                        "id": variable,
                        "type": "string",
                        "operator": "is_not_null",
                        "value": None,
                    }
                    for variable in {"lefthippocampus"}
                ],
                "valid": True,
            },
        },
        "parameters": None,
        "test_case_num": 99,
    }
    input["type"] = "exaflow"
    algorithm_result = get_algorithm_result("standard_deviation", input)
    print(algorithm_result)

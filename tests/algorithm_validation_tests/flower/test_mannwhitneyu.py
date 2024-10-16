def test_mann_whitney(get_algorithm_result):
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
            "validation_datasets": ["ppmi_test"],
            "filters": None,
        },
        "parameters": None,
        "test_case_num": 100,
    }
    input["type"] = "flower"
    algorithm_result = get_algorithm_result("mann_whitneyu", input)

    # Extract p_value from the returned result
    assert algorithm_result["metrics_aggregated"]
    print(algorithm_result["metrics_aggregated"])

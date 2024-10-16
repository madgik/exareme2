def test_chi_squared(get_algorithm_result):
    input = {
        "inputdata": {
            "y": ["alzheimerbroadcategory"],
            "x": ["agegroup"],
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
        "test_case_num": 101,
    }
    input["type"] = "flower"
    algorithm_result = get_algorithm_result("chi_squared", input)

    # Extract p_value from the returned result
    metrics_aggregated = algorithm_result["metrics_aggregated"]
    assert metrics_aggregated
    print(metrics_aggregated)

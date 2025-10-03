from tests.algorithm_validation_tests.exaflow.helpers import algorithm_request
from tests.algorithm_validation_tests.exaflow.helpers import parse_response


def test_logistic_regression():
    input = {
        "inputdata": {
            "y": ["alzheimerbroadcategory"],
            "x": ["leftsogsuperioroccipitalgyrus", "rightmfcmedialfrontalcortex"],
            "data_model": "dementia:0.1",
            "datasets": [
                "desd-synthdata8",
                "desd-synthdata0",
                "ppmi0",
                "edsd8",
                "desd-synthdata5",
                "ppmi7",
                "desd-synthdata1",
                "edsd4",
                "desd-synthdata4",
                "edsd2",
                "ppmi6",
                "ppmi9",
            ],
        },
        "parameters": {"positive_class": "AD"},
    }
    response = algorithm_request("exaflow_logistic_regression", input)
    result = parse_response(response)
    print(result)

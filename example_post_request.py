import requests
from mipengine.controller.api.algorithm_request_dto import (
    AlgorithmInputDataDTO,
    AlgorithmRequestDTO,
)

from devtools import debug


def do_post_request():
    url = "http://127.0.0.1:5000/algorithms" + "/logistic_regression"

    pathology = "dementia"
    datasets = ["edsd"]
    x = [
        "lefthippocampus",
        "righthippocampus",
        "rightppplanumpolare",
        "leftamygdala",
        "rightamygdala",
    ]
    y = ["alzheimerbroadcategory"]
    filters = {
        "condition": "AND",
        "rules": [
            {
                "id": "dataset",
                "type": "string",
                "value": datasets,
                "operator": "in",
            },
            {
                "condition": "AND",
                "rules": [
                    {
                        "id": variable,
                        "type": "string",
                        "operator": "is_not_null",
                        "value": None,
                    }
                    for variable in x + y
                ],
            },
        ],
        "valid": True,
    }
    classes = ["AD", "CN"]

    algorithm_input_data = AlgorithmInputDataDTO(
        pathology=pathology,
        datasets=datasets,
        filters=filters,
        x=x,
        y=y,
    )

    algorithm_request = AlgorithmRequestDTO(
        inputdata=algorithm_input_data,
        parameters={"classes": classes},
    )

    debug(algorithm_request)
    print(f"POSTing to {url}")

    request_json = algorithm_request.json()

    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    response = requests.post(url, data=request_json, headers=headers)

    return response


if __name__ == "__main__":
    response = do_post_request()
    print(f"Algorithm result-> {response.text}")

import requests
from mipengine.controller.api.AlgorithmRequestDTO import (
    AlgorithmInputDataDTO,
    AlgorithmRequestDTO,
)


def do_post_request():
    url = "http://127.0.0.1:5000/algorithms" + "/logistic_regression"

    x = [
        "lefthippocampus",
        "righthippocampus",
        "rightppplanumpolare",
        "leftamygdala",
        "rightamygdala",
    ]
    y = ["alzheimerbroadcategory"]
    classes = ["AD", "CN"]

    pathology = "dementia"
    datasets = ["edsd"]

    print(f"POST to {url}")
    print(f"X: {x}")
    print(f"y: {y}")
    print(f"Target classes: {classes}")
    print(f"Pathology: {pathology}, datasets: {datasets}")

    filters = {
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
        "valid": True,
    }

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

    request_json = algorithm_request.to_json()

    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    response = requests.post(url, data=request_json, headers=headers)

    return response


if __name__ == "__main__":
    response = do_post_request()
    print(f"Algorithm result-> {response.text}")

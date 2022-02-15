import json

import requests
from devtools import debug

from mipengine.controller.api.algorithm_request_dto import AlgorithmInputDataDTO
from mipengine.controller.api.algorithm_request_dto import AlgorithmRequestDTO


def do_post_request():
    url = "http://127.0.0.1:5000/algorithms" + "/pca"

    data_model = "dementia:0.1"
    datasets = ["edsd"]
    x = [
        "lefthippocampus",
        "righthippocampus",
        "rightppplanumpolare",
        "leftamygdala",
        "rightamygdala",
    ]
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
                    for variable in x
                ],
            },
        ],
        "valid": True,
    }

    algorithm_input_data = AlgorithmInputDataDTO(
        data_model=data_model,
        datasets=datasets,
        filters=filters,
        x=x,
    )

    algorithm_request = AlgorithmRequestDTO(
        inputdata=algorithm_input_data,
        parameters={},
    )

    print(f"POSTing to {url}:")
    debug(algorithm_request)

    request_json = algorithm_request.json()

    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    response = requests.post(url, data=request_json, headers=headers)

    return response


if __name__ == "__main__":
    response = do_post_request()
    print("\nResponse:")
    print(f"{response.status_code=}")
    print(f"Result={json.dumps(json.loads(response.text), indent=4)}")

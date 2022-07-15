import json

import requests
from devtools import debug

from mipengine.controller.api.algorithm_request_dto import AlgorithmInputDataDTO
from mipengine.controller.api.algorithm_request_dto import AlgorithmRequestDTO


def do_post_request():
    url = "http://127.0.0.1:5000/algorithms" + "/linear_regression_cv"

    x = [
        "rightmfcmedialfrontalcortex",
        "rightsogsuperioroccipitalgyrus",
        "ppmicategory",
    ]
    y = ["leftcerebellumwhitematter"]
    data_model = "dementia:0.1"
    datasets = [
        "desd-synthdata5",
        "edsd1",
        "desd-synthdata2",
        "desd-synthdata4",
        "desd-synthdata6",
        "ppmi4",
        "edsd9",
        "desd-synthdata0",
        "edsd7",
        "ppmi6",
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
                    for variable in x + y
                ],
            },
        ],
        "valid": True,
    }
    parameters = {"n_splits": 17}

    algorithm_input_data = AlgorithmInputDataDTO(
        data_model=data_model,
        datasets=datasets,
        filters=filters,
        x=x,
        y=y,
    )

    algorithm_request = AlgorithmRequestDTO(
        inputdata=algorithm_input_data,
        parameters=parameters,
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
    try:
        print(f"Result={json.dumps(json.loads(response.text), indent=4)}")
    except json.decoder.JSONDecodeError:
        print(f"Something went wrong:\n{response.text}")

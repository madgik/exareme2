import json

import requests
from devtools import debug

from mipengine.controller.api.algorithm_request_dto import AlgorithmInputDataDTO
from mipengine.controller.api.algorithm_request_dto import AlgorithmRequestDTO


def do_post_request():
    url = "http://127.0.0.1:5000/algorithms" + "/logistic_regression"

    data_model = "dementia:0.1"
    datasets = [
        "ppmi0",
        "ppmi1",
        "ppmi2",
        "ppmi3",
        "ppmi4",
        "ppmi5",
        "ppmi6",
        "ppmi7",
        "ppmi8",
        "ppmi9",
    ]
    x = [
        "righthippocampus",
        "alzheimerbroadcategory",
        # "agegroup"
        # "leftphgparahippocampalgyrus",
        # "rightpallidum",
        # "leftsmcsupplementarymotorcortex",
        # "leftcalccalcarinecortex"
    ]
    y = [
        "ppmicategory",
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

    algorithm_input_data = AlgorithmInputDataDTO(
        data_model=data_model,
        datasets=datasets,
        filters=filters,
        x=x,
        y=y,
    )

    algorithm_request = AlgorithmRequestDTO(
        inputdata=algorithm_input_data,
        parameters={"positive_class": "GENPD"},
    )

    request_json = algorithm_request.json()

    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    response = requests.post(url, data=request_json, headers=headers)

    return response


if __name__ == "__main__":
    response = do_post_request()
    try:
        print(f"{json.dumps(json.loads(response.text), indent=4)}")
    except json.decoder.JSONDecodeError:
        print(f"Something went wrong:\n{response.text}")

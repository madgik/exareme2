import requests
from devtools import debug

from mipengine.controller.api.algorithm_request_dto import AlgorithmInputDataDTO
from mipengine.controller.api.algorithm_request_dto import AlgorithmRequestDTO


def do_post_request():
    url = "http://127.0.0.1:5000/algorithms" + "/descriptive_statistics"

    data_model = "dementia:0.1"
    datasets = [
        "edsd0",
        "edsd1",
        "edsd2",
        "edsd3",
        "edsd4",
        "edsd5",
        "edsd6",
        "edsd7",
        "edsd8",
        "edsd9",
    ]
    x = [
        "lefthippocampus",
        "righthippocampus",
        "rightppplanumpolare",
        "leftamygdala",
        "rightamygdala",
    ]
    y = ["parkinsonbroadcategory", "neurodegenerativescategories", "adnicategory"]
    filters = {
        "condition": "AND",
        "rules": [
            {
                "id": "dataset",
                "type": "string",
                "value": datasets,
                "operator": "in",
            },
        ],
        "valid": True,
    }
    # classes = ["AD", "CN"]

    algorithm_input_data = AlgorithmInputDataDTO(
        data_model=data_model,
        datasets=datasets,
        filters=filters,
        x=x,
        y=y,
    )

    algorithm_request = AlgorithmRequestDTO(
        inputdata=algorithm_input_data,
        # parameters={"classes": classes},
    )

    debug(algorithm_request)
    print(f"POSTing to {url}")

    request_json = algorithm_request.json()

    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    response = requests.post(url, data=request_json, headers=headers)

    return response


if __name__ == "__main__":
    response = do_post_request()
    print(f"\nResponse:")
    print(f"Status code-> {response.status_code}")
    print(f"Algorithm result-> {response.text}")

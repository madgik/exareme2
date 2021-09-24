import requests

from mipengine.algorithm_request_DTO import AlgorithmRequestDTO


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
    algorithm_params = {"classes": ["AD", "CN"]}

    algorithm_request_dto = AlgorithmRequestDTO(
        pathology=pathology,
        datasets=datasets,
        x=x,
        y=y,
        filters=filters,
        algorithm_params=algorithm_params,
    )

    # DEBUG
    print(f"POST to {url}")
    # print("algorithm_request_dto:")
    # import pprint
    # pprint.PrettyPrinter(indent=0, sort_dicts=False).pprint(
    #     algorithm_request_dto.dict()
    # )

    from devtools import debug

    debug(algorithm_request_dto)
    # DEBUG end

    algorithm_request_dto_json = algorithm_request_dto.json()

    # # DEBUG
    # import json
    # formatted = json.dumps(json.loads(algorithm_request_dto_json), indent=2)
    # print("\nalgorithm_request_dto_json:")
    # print(formatted)
    # # DEBUG end

    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    response = requests.post(url, data=algorithm_request_dto_json, headers=headers)

    return response


if __name__ == "__main__":
    response = do_post_request()
    print(f"Algorithm result-> {response.text}")

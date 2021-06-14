import requests
from mipengine.controller.api.AlgorithmRequestDTO import (
    AlgorithmInputDataDTO,
    AlgorithmRequestDTO,
)


def do_post_request():
    url = "http://127.0.0.1:5000/algorithms" + "/logistic_regression"
    print(f"POST to {url}")

    # example request 1
    # algorithm_input_data = AlgorithmInputDataDTO(pathology="mentalhealth",
    #                                              datasets=["demo"],
    #                                              filters=None,
    #                                              x=["rightaccumbensarea", "leftaccumbensarea", "rightamygdala"],
    #                                              y=["leftamygdala"])

    # example request 2
    algorithm_input_data = AlgorithmInputDataDTO(
        pathology="dementia",
        datasets=["demo_data"],
        filters=None,
        x=["lefthippocampus", "righthippocampus"],
        y=["alzheimerbroadcategory_bin"],
    )

    algorithm_request = AlgorithmRequestDTO(
        inputdata=algorithm_input_data, parameters=None
    )

    request_json = algorithm_request.to_json()

    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    algorithm_result = requests.post(url, data=request_json, headers=headers)

    return algorithm_result


result = do_post_request()

print(f"Algorithm result-> {result.text}")

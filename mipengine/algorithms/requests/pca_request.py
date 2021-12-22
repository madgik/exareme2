import requests
import json


def do_post_request(input):
    url = "http://127.0.0.1:5000/algorithms" + "/pca"

    filters = {
        "condition": "AND",
        "rules": [
            {
                "id": "dataset",
                "type": "string",
                "value": input["inputdata"]["datasets"],
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
                    for variable in input["inputdata"]["x"]
                ],
            },
        ],
        "valid": True,
    }
    input["inputdata"]["filters"] = filters
    request_json = json.dumps(input)

    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    response = requests.post(url, data=request_json, headers=headers)
    return response


# if __name__ == "__main__":
#     response = do_post_request(input)
#     print(f"\nResponse:")
#     print(f"Status code-> {response.status_code}")
#     print(f"Algorithm result-> {response.text}")

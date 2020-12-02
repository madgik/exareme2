import requests

url="http://localhost:9999"

json={
    "algorithm": {
        "files":{
            "algorithmFolder":"kmeans",
            "algorithmFlowFile": "kmeans_flow",
            "algorithmUDFsFile":"kmeans_udfs"
        },
        "parameters":{
            "numOfClusters": 3,
            "numOfIterations":10,
            "initialMin":-10,
            "initialMax":10
        }
    },
    "data": {
        "table": "data",
        "attributes": ["c1", "c2"],
        "filters": [
            ["c2", ">", "2"],
            ["c1", "<", "9"]
        ]
    }
}

result=requests.post(url,json=json)
print(result.text)

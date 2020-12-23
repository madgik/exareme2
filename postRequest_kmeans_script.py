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
            "numOfClusters": 5,
            "numOfIterations":10,
            "initialMin":-10,
            "initialMax":10
        }
    },
    "data": {
        "table": "data3",
        "attributes": ["v1","v2"],
        "filters": [
            ["c2", ">", "2"],
            ["c1", "<", "9"]
        ]
    }
}

result=requests.post(url,json=json)
print(result.text)

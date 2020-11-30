import requests

url="http://localhost:9999"

#data={'algorithm':'kmeans',
#        'params':'{"table":"data",\
#                    "attributes":["c1","c2"],\
#                    "parameters":[0.7,4],\
#                    "filters":[\
#                                [\
#                                    ["c1",">","2"],\
#                                    ["c1","<","10000"]\
#                                ],\
#                                [\
#                                    ["c1",">","0"]\
#                                ]\
#                            ]\
#                    }'
#    }

json={
    "algorithmParams": {
        "algorithmFolder":"kmeans",
        "algorithmFlowFile": "kmeans_flow",
        "algorithmUDFsFile":"kmeans",
        "numOfClusters": 3,
        "numOfIterations":10,
        "initialMin":-10,
        "initialMax":10
    },
    "dataParams": {
        "table": "data",
        "attributes": ["c1", "c2"],
        "filters": [
            ["c2", ">", "2"],
            ["c1", "<", "9"],
        ]
    }
}

result=requests.post(url,json=json)
print(result.text)

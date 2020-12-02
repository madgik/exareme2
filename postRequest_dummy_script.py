import requests

url="http://localhost:9999"

json={
    "algorithm": {
        "files":{
            "algorithmFolder":"dummy",
            "algorithmFlowFile": "dummy_flow",
            "algorithmUDFsFile":"dummy_udfs"
        },
        "parameters":{
        }
    },
    "data": {
        "table": "data",
        "attributes": [],
        "filters": []
    }
}


result=requests.post(url,json=json)
print(result.text)

import requests

url="http://localhost:9999"

json={
    "algorithmParams": {
        "algorithmFolder":"dummy",
        "algorithmFlowFile": "dummy_flow",
        "algorithmUDFsFile":"dummy_udfs"
    },
    "dataParams": {
        "table": "data",
        "attributes": [],
        "filters": []
    }
}

result=requests.post(url,json=json)
print(result.text)

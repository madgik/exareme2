import requests

url="http://localhost:9999"

json={
    "algorithmParams": {
        "algorithmFolder":"dummy",
        "algorithmFlowFile": "dummy_flow",
        "algorithmUDFsFile":"dummy_udfs",
        "input_par1": 1.234,
        "input_par2":5.678
    },
    "dataParams": {
        "table": "data",
        "attributes": [],
        "filters": []
    }
}

result=requests.post(url,json=json)
print(result.text)

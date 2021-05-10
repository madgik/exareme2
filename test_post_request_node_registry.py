from mipengine.common.node_registry_DTOs import Pathology, NodeRecord, NodeRecordsList
from ipaddress import IPv4Address
import requests


def do_post_request():
    a_pathology = Pathology(name="pathology1", datasets=["dataset1", "dataset2"])
    a_node_record_1 = NodeRecord(
        node_id="node1",
        task_queue_ip=IPv4Address("127.0.0.123"),
        task_queue_port=1234,
        db_ip=IPv4Address("127.0.0.1"),
        db_queue_port=5678,
        pathologies=[a_pathology],
    )
    a_node_record_1_json = a_node_record_1.json()

    a_node_record_2 = NodeRecord(
        node_id="node2",
        task_queue_ip=IPv4Address("127.0.0.124"),
        task_queue_port=1234,
        db_ip=IPv4Address("127.0.0.2"),
        db_queue_port=5678,
        pathologies=[a_pathology],
    )
    a_node_record_2_json = a_node_record_2.json()

    url = "http://127.0.0.1:5000/register"
    headers = {"Content-type": "application/json", "Accept": "text/plain"}

    # register node1
    result = requests.post(url, data=a_node_record_1_json, headers=headers)
    if result.status_code == requests.codes.ok:
        print(f"\nADDED: \n{a_node_record_1_json}")
    elif result.status_code == requests.codes.conflict:
        print(f"\nFAILED TO ADD: \n{a_node_record_1_json} \nREASON: {result.text}")

    # register node2
    result = requests.post(url, data=a_node_record_2_json, headers=headers)
    if result.status_code == requests.codes.ok:
        print(f"\nADDED: \n{a_node_record_2_json}")
    elif result.status_code == requests.codes.conflict:
        print(f"\nFAILED TO ADD: \n{a_node_record_2_json} \nREASON: {result.text}")

    # get all registered nodes
    url = "http://127.0.0.1:5000/nodes"
    result = requests.get(url)

    return result


result = do_post_request()

node_records_list = NodeRecordsList.parse_raw(result.text)
print("\nREGISTERED NODES:")
[print(record) for record in node_records_list.node_records]

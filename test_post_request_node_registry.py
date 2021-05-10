from mipengine.common.node_catalog_DTOs import Pathology, NodeRecord, NodeRecordsList
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
    print(f"POST to {url}")

    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    result = requests.post(url, data=a_node_record_1_json, headers=headers)
    print(f"node_record1 register result -> {result}")

    result = requests.post(url, data=a_node_record_2_json, headers=headers)
    print(f"node_record2 register result -> {result.text}")

    # get_all_nodes
    url = "http://127.0.0.1:5000/nodes"
    result = requests.get(url)

    print(f"get_all_nodes result -> {result}")
    return result


result = do_post_request()

# print(f"FINAL result-> \n{result.text}")
node_records_list = NodeRecordsList.parse_raw(result.text)
# print(f"\n\n{node_records_list=}")
[print(record) for record in node_records_list.node_records]

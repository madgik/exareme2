import mipengine.service_registry.service_registry as sr
from mipengine.common.node_registry_DTOs import NodeRecord, Pathology, NodeRole
from ipaddress import IPv4Address

a_pathology1 = Pathology(name="pathology1", datasets=["dataset1", "dataset2"])
a_node_record_1 = NodeRecord(
    node_id="node1",
    node_role=NodeRole.GLOBAL_NODE,
    task_queue_ip=IPv4Address("127.0.0.123"),
    task_queue_port=1234,
    db_id="node1_db",
    db_ip=IPv4Address("127.0.0.1"),
    db_port=5678,
    pathologies=[a_pathology1],
)

a_pathology2 = Pathology(name="pathology2", datasets=["dataset3", "dataset4"])
a_node_record_2 = NodeRecord(
    node_id="node2",
    node_role=NodeRole.LOCAL_NODE,
    task_queue_ip=IPv4Address("127.0.0.124"),
    task_queue_port=1234,
    db_id="node2_db",
    db_ip=IPv4Address("127.0.0.2"),
    db_port=5678,
    pathologies=[a_pathology2],
)

# register 2 nodes
sr.register_node(a_node_record_1)
sr.register_node(a_node_record_2)

# get all registered nodes 
all_nodes=sr.get_all_nodes()
print("get_all_nodes() returns:")
{print(f"{node_id=}\n{node_info=}") for (node_id,node_info) in all_nodes.items()}

# when a node is registered, its database gets also registered
# get info about the db associated with this node
try:
    db_info=sr.get_db_info("node1")
    print(f"\n{db_info=}")
except sr.NodeIDNotInKVStore as exc:
    print(f"{exc.message=}")

# deregister a node
try:
    sr.deregister_node("node1")
except sr.NodeIDNotInKVStore as exc:
    print(f"{exc.message=}")

# deregistering a node deregisters the associated db as well
try:
    db_info=sr.get_db_info("node1")
    print(f"\n{db_info=}")
except sr.NodeIDNotInKVStore as exc:
    print(f"{exc.message=}")
    

all_nodes=sr.get_all_nodes()
print("get_all_nodes() returns:")
{print(f"{node_id=}\n{node_info=}") for (node_id,node_info) in all_nodes.items()}

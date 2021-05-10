import pytest
from ipaddress import IPv4Address
from mipengine.common.node_registry_DTOs import Pathology, NodeRecord, NodeRecordsList
from mipengine.node_registry.node_registry import NodeRegistry


@pytest.fixture
def a_pathology():
    a_pathology = Pathology(name="pathology1", datasets=["dataset1", "dataset2"])
    return a_pathology


@pytest.fixture
def a_node_record1(a_pathology):
    a_node_record = NodeRecord(
        node_id="node1",
        task_queue_ip=IPv4Address("127.0.0.123"),
        task_queue_port=1234,
        db_ip=IPv4Address("127.0.0.1"),
        db_queue_port=5678,
        pathologies=[a_pathology],
    )
    return a_node_record


@pytest.fixture
def a_node_record2(a_pathology):
    a_node_record = NodeRecord(
        node_id="node2",
        task_queue_ip=IPv4Address("127.0.0.124"),
        task_queue_port=5678,
        db_ip=IPv4Address("127.0.0.2"),
        db_queue_port=9101,
        pathologies=[a_pathology],
    )
    return a_node_record


@pytest.fixture
def empty_node_registry():
    return NodeRegistry()


@pytest.fixture
async def filled_node_registry(a_node_record1, a_node_record2):
    filled_node_registry = NodeRegistry()
    await filled_node_registry.register_node(a_node_record1)
    await filled_node_registry.register_node(a_node_record2)
    return filled_node_registry


@pytest.mark.asyncio
async def test_register_node(empty_node_registry, a_node_record1, a_node_record2):
    print(f"{a_node_record1=}")
    await empty_node_registry.register_node(a_node_record1)
    await empty_node_registry.register_node(a_node_record2)
    the_nodes = await empty_node_registry.get_all_nodes()
    pytest.assume(a_node_record1 in the_nodes.node_records)
    pytest.assume(a_node_record2 in the_nodes.node_records)


@pytest.mark.asyncio
async def test_register_same_node_id(empty_node_registry, a_node_record1):
    await empty_node_registry.register_node(a_node_record1)
    import copy

    a_node_record1_changed = copy.deepcopy(a_node_record1)
    a_node_record1_changed.task_queue_ip = IPv4Address("1.1.1.1")
    a_node_record1_changed.db_ip = IPv4Address("1.1.1.1")

    with pytest.raises(NodeRegistry.NodeIdAlreadyInRegistryError) as e:
        await empty_node_registry.register_node(a_node_record1_changed)


@pytest.mark.asyncio
async def test_register_same_task_queue_ip(empty_node_registry, a_node_record1):
    await empty_node_registry.register_node(a_node_record1)

    import copy

    a_node_record1_changed = copy.deepcopy(a_node_record1)
    a_node_record1_changed.node_id = "a_different_id"
    a_node_record1_changed.db_ip = IPv4Address("1.1.1.1")

    with pytest.raises(NodeRegistry.TaskQueueIPAlreadyInUseError) as e:
        await empty_node_registry.register_node(a_node_record1_changed)


@pytest.mark.asyncio
async def test_register_same_node_id(empty_node_registry, a_node_record1):
    await empty_node_registry.register_node(a_node_record1)

    import copy

    a_node_record1_changed = copy.deepcopy(a_node_record1)
    a_node_record1_changed.node_id = "a_different_id"
    a_node_record1_changed.task_queue_ip = IPv4Address("1.1.1.1")

    with pytest.raises(NodeRegistry.DatabaseIPAlreadyInUseError) as e:
        await empty_node_registry.register_node(a_node_record1_changed)


@pytest.mark.asyncio
async def test_deregister_node(filled_node_registry, a_node_record1):
    await filled_node_registry.deregister_node("node1")
    the_nodes = await filled_node_registry.get_all_nodes()
    pytest.assume(a_node_record1 not in the_nodes.node_records)


@pytest.mark.skip
@pytest.mark.asyncio
async def test_update_node(filled_node_registry, a_node_record1):
    a_node_record1[1].task_queue_ip = IPv4Address("127.0.0.1")
    await filled_node_registry.update_node("node1", a_node_record1[1])
    tmp = await filled_node_registry.get_node("node1")
    assert tmp == a_node_record1[1]

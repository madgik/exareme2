from mipengine.udfgen import make_unique_func_name
from tests.integration_tests import nodes_communication

local_node_id = "localnode1"
command_id = "command123"
localnode_app = nodes_communication.get_celery_app(local_node_id)
local_node_get_udf = nodes_communication.get_celery_task_signature(
    localnode_app, "get_udf"
)
local_node_get_run_udf_query = nodes_communication.get_celery_task_signature(
    localnode_app, "get_run_udf_query"
)
local_node_run_udf = nodes_communication.get_celery_task_signature(
    localnode_app, "run_udf"
)
local_node_create_table = nodes_communication.get_celery_task_signature(
    localnode_app, "create_table"
)
local_node_cleanup = nodes_communication.get_celery_task_signature(
    localnode_app, "clean_up"
)


# @pytest.fixture()
# def context_id():
#     context_id = "test_udfs_" + uuid.uuid4().hex
#
#     yield context_id
#
#     local_node_cleanup.delay(context_id=context_id).get()


def test_get_udf():
    from tests.algorithms.count_rows import get_column_rows

    fetched_udf = local_node_get_udf.delay(
        func_name=make_unique_func_name(get_column_rows)
    ).get()

    assert get_column_rows.__name__ in fetched_udf

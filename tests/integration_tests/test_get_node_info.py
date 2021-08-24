import json

import pytest

from mipengine.node_info_DTOs import NodeInfo
from tests.integration_tests.nodes_communication import get_celery_app
from tests.integration_tests.nodes_communication import get_celery_task_signature


test_cases_get_node_info = [
    (
        "globalnode",
        NodeInfo(
            id="globalnode",
            role="GLOBALNODE",
            ip="172.17.0.1",
            port=5670,
            db_ip="172.17.0.1",
            db_port=50000,
            datasets_per_schema=None,
        ),
    ),
    (
        "localnode1",
        NodeInfo(
            id="localnode1",
            role="LOCALNODE",
            ip="172.17.0.1",
            port=5671,
            db_ip="172.17.0.1",
            db_port=50001,
            datasets_per_schema={
                "dementia": [
                    "edsd",
                    "ppmi",
                    "desd-synthdata",
                    "fake_longitudinal",
                    "demo_data",
                ],
                "mentalhealth": ["demo"],
                "tbi": ["tbi_demo2"],
            },
        ),
    ),
]


@pytest.mark.parametrize(
    "node_id, node_info",
    test_cases_get_node_info,
)
def test_get_node_info(node_id, node_info):
    node_app = get_celery_app(node_id)
    node_info_signature = get_celery_task_signature(node_app, "get_node_info")

    assert node_info_signature.delay().get() == node_info.json()

from unittest import mock
from unittest.mock import patch

import pytest

from mipengine.controller.controller import DataModelViewsCreator
from mipengine.controller.controller import _data_model_views_to_localnodestables
from mipengine.controller.controller import _validate_number_of_views


@pytest.fixture
def views_per_local_nodes():
    tables_views = {
        "node1": ["view1_1", "view1_2"],
        "node2": ["view2_1", "view2_2"],
        "node3": ["view3_1", "view3_2"],
    }
    return tables_views


@pytest.fixture
def nodes_tables_expected():
    expected = [
        {"node1": "view1_1", "node2": "view2_1", "node3": "view3_1"},
        {"node1": "view1_2", "node2": "view2_2", "node3": "view3_2"},
    ]
    return expected


@pytest.fixture
def views_per_local_nodes_invalid():
    tables_views = {
        "node1": ["view1_1", "view1_2"],
        "node2": ["view2_1"],
        "node3": ["view3_1", "view3_2"],
    }
    return tables_views


def test_validate_number_of_views(views_per_local_nodes, views_per_local_nodes_invalid):
    assert _validate_number_of_views(views_per_local_nodes) == len(
        list(views_per_local_nodes.values())[0]
    )

    with pytest.raises(ValueError):
        _validate_number_of_views(views_per_local_nodes_invalid)


def test_data_model_views_to_localnodestables(
    views_per_local_nodes, nodes_tables_expected
):
    class MockLocalNodesTable:
        def __init__(self, nodes_tables_info: dict):
            self._nodes_tables_info = nodes_tables_info

    with patch(
        "mipengine.controller.controller.LocalNodesTable",
        MockLocalNodesTable,
    ):
        local_nodes_tables = _data_model_views_to_localnodestables(
            views_per_local_nodes
        )
        nodes_tables_info = [t._nodes_tables_info for t in local_nodes_tables]
        for expected in nodes_tables_expected:
            assert expected in nodes_tables_info

        assert len(nodes_tables_expected) == len(nodes_tables_info)

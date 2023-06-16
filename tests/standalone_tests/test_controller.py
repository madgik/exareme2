import functools
import itertools
import unittest.mock
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from unittest.mock import patch

import pytest

from mipengine import DType
from mipengine.controller.algorithm_execution_engine_tasks_handler import (
    INodeAlgorithmTasksHandler,
)
from mipengine.controller.algorithm_flow_data_objects import LocalNodesTable
from mipengine.controller.controller import DataModelViews
from mipengine.controller.controller import DataModelViewsCreator
from mipengine.controller.controller import DataModelViewsCreatorInitParams
from mipengine.controller.controller import _data_model_views_to_localnodestables
from mipengine.controller.controller import _validate_number_of_views
from mipengine.controller.nodes import LocalNode
from mipengine.exceptions import InsufficientDataError
from mipengine.node_tasks_DTOs import ColumnInfo
from mipengine.node_tasks_DTOs import NodeUDFDTO
from mipengine.node_tasks_DTOs import NodeUDFKeyArguments
from mipengine.node_tasks_DTOs import NodeUDFPosArguments
from mipengine.node_tasks_DTOs import TableData
from mipengine.node_tasks_DTOs import TableInfo
from mipengine.node_tasks_DTOs import TableSchema
from mipengine.node_tasks_DTOs import TableType


def create_mock_node(node_id: str, context_id: str, request_id: str):
    return LocalNode(
        request_id=request_id,
        context_id=context_id,
        node_tasks_handler=NodeAlgorithmTasksHandlerMock(node_id),
    )


@pytest.fixture
def node_mocks():
    context_id = "0"
    nodes_ids = ["node" + str(i) for i in range(1, 11)]
    nodes = [
        create_mock_node(
            request_id="0",
            context_id=context_id,
            node_tasks_handler=NodeAlgorithmTasksHandlerMock(node_id),
        )
        for node_id in nodes_ids
    ]
    return nodes


class TestDataModelViews:
    @pytest.fixture
    def views_mocks(self):

        schema = TableSchema(
            columns=[
                ColumnInfo(name="col_1", dtype=DType.INT),
                ColumnInfo(name="col_2", dtype=DType.INT),
                ColumnInfo(name="col_3", dtype=DType.INT),
            ]
        )

        nodes_table_info_1 = {}
        expected_nodes_1 = []
        for i in range(1, 5):
            context_id = "0"
            local_node = create_mock_node(
                node_id="node" + str(i),
                request_id="0",
                context_id=context_id,
            )
            expected_nodes_1.append(local_node)

            table_type = TableType.NORMAL

            # table naming convention <table_type>_<node_id>_<context_id>_<command_id>_<result_id>
            table_info = TableInfo(
                name=str(table_type).lower()
                + "_"
                + local_node.node_id
                + "_0_"
                + context_id
                + "_0",
                schema_=schema,
                type_=TableType.NORMAL,
            )
            nodes_table_info_1[local_node] = table_info

        local_nodes_table_1 = LocalNodesTable(nodes_tables_info=nodes_table_info_1)

        nodes_table_info_2 = {}
        expected_nodes_2 = []
        for i in range(3, 7):
            context_id = "1"
            local_node = create_mock_node(
                node_id="node" + str(i),
                request_id="0",
                context_id=context_id,
            )
            expected_nodes_2.append(local_node)

            table_type = TableType.NORMAL

            # table naming convention <table_type>_<node_id>_<context_id>_<command_id>_<result_id>
            table_info = TableInfo(
                name=str(table_type).lower()
                + "_"
                + local_node.node_id
                + "_0_"
                + context_id
                + "_0",
                schema_=schema,
                type_=TableType.NORMAL,
            )
            nodes_table_info_2[local_node] = table_info

        local_nodes_table_2 = LocalNodesTable(nodes_tables_info=nodes_table_info_2)

        return [
            (local_nodes_table_1, expected_nodes_1),
            (local_nodes_table_2, expected_nodes_2),
        ]

    def test_get_nodes(self, views_mocks):
        views = [t[0] for t in views_mocks]

        expected_nodes = [t[1] for t in views_mocks]
        expected_nodes_flat = list(itertools.chain(*expected_nodes))

        data_model_views = DataModelViews(views)
        nodes = data_model_views.get_nodes()

        assert all(item in nodes for item in expected_nodes_flat)


class TestDataModelViewsCreator:
    @pytest.fixture
    def node_mocks(self):
        context_id = "0"
        nodes_ids = ["node" + str(i) for i in range(5)]
        nodes = [
            create_mock_node(node_id=node_id, context_id=context_id, request_id="0")
            for node_id in nodes_ids
        ]
        return nodes

    @pytest.fixture
    def nodes_datasets_mock(self, node_mocks):
        nodes_datasets_mock = {
            node_mocks[0]: ["dataset1", "dataset2", "dataset6"],
            node_mocks[1]: ["dataset2", "dataset3", "dataset6"],
            node_mocks[2]: ["dataset7", "dataset6", "dataset9", "dataset2"],
            node_mocks[3]: ["dataset6", "dataset1", "dataset3", "dataset8", "dataset2"],
            node_mocks[4]: ["dataset2", "dataset6"],
        }

        has_node_sufficient_data = [
            True,
            True,
            False,
            True,
            True,
            False,
            True,
            True,
            True,
            False,
        ]
        sufficient_data_nodes = dict(zip(node_mocks, has_node_sufficient_data))

        def create_data_model_views_mock(self, *args, **kwargs) -> str:
            if sufficient_data_nodes[self]:
                return "a_dummy_view_table"
            else:
                raise InsufficientDataError(f"Node:{self.node_id} has not enough data")

        for node in nodes_datasets_mock.keys():
            node.create_data_model_views = functools.partial(
                create_data_model_views_mock, node
            )

        return (nodes_datasets_mock, sufficient_data_nodes)

    # check that  node.create_data_model_views is called with the relevant params for all nodes
    # check it returns the "correct" DataModelViews obj
    def test_create_data_model_views(self, nodes_datasets_mock):
        nodes_datasets_mock, sufficient_data_nodes = nodes_datasets_mock
        request_datasets = ["dataset2", "dataset6"]

        init_params = DataModelViewsCreatorInitParams(
            nodes_datasets=nodes_datasets_mock,
            data_model="datamodel",
            datasets=request_datasets,
            variable_groups=[],
            var_filters=[],
            dropna=True,
            check_min_rows=True,
            command_id=0,
        )
        with unittest.mock.patch(
            "mipengine.controller.controller._data_model_views_to_localnodestables",
            new=_data_model_views_to_localnodestables_mock,
        ):
            data_model_views_creator = DataModelViewsCreator(init_params)
            data_model_views_creator.create_data_model_views()

            expected = [
                node.node_id
                for node, has_sufficient_data in sufficient_data_nodes.items()
                if has_sufficient_data
            ]

            assert set(expected) == set(
                data_model_views_creator.data_model_views.to_list()
            )

    # check InsufficientDataError if not enough data on any of the nodes
    def test_create_data_model_views_insufficient_data(self, nodes_datasets_mock):
        nodes_datasets_mock, __ = nodes_datasets_mock

        def create_data_model_views_mock(self, *args, **kwargs) -> str:
            raise InsufficientDataError(f"Node:{self.node_id} has not enough data")

        for node in nodes_datasets_mock.keys():
            node.create_data_model_views = functools.partial(
                create_data_model_views_mock, node
            )

        request_datasets = ["dataset2", "dataset6"]

        init_params = DataModelViewsCreatorInitParams(
            nodes_datasets=nodes_datasets_mock,
            data_model="datamodel",
            datasets=request_datasets,
            variable_groups=[],
            var_filters=[],
            dropna=True,
            check_min_rows=True,
            command_id=0,
        )
        data_model_views_creator = DataModelViewsCreator(init_params)
        with pytest.raises(InsufficientDataError):
            data_model_views_creator.create_data_model_views()


def _data_model_views_to_localnodestables_mock(
    views_per_localnode: Dict[LocalNode, List[TableInfo]]
) -> List[str]:  # List[LocalNodesTable]:
    # only the node_id portion of the dict is used in the test
    return [node.node_id for node in views_per_localnode.keys()]


class AsyncResult:
    pass


class NodeAlgorithmTasksHandlerMock(INodeAlgorithmTasksHandler):
    def __init__(self, node_id: str):
        self._node_id = node_id

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def node_data_address(self) -> str:
        pass

    @property
    def tasks_timeout(self) -> int:
        pass

    # TABLES functionality
    def get_tables(self, request_id: str, context_id: str) -> List[str]:
        pass

    def get_table_data(self, request_id: str, table_name: str) -> TableData:
        pass

    def create_table(
        self, request_id: str, context_id: str, command_id: str, schema: TableSchema
    ) -> TableInfo:
        pass

    # VIEWS functionality
    def get_views(self, request_id: str, context_id: str) -> List[str]:
        pass

    def create_data_model_views(
        self,
        request_id: str,
        context_id: str,
        command_id: str,
        data_model: str,
        datasets: List[str],
        columns_per_view: List[List[str]],
        filters: dict,
        dropna: bool = True,
        check_min_rows: bool = True,
    ) -> List[TableInfo]:
        pass

    # MERGE TABLES functionality
    def get_merge_tables(self, request_id: str, context_id: str) -> List[str]:
        pass

    def create_merge_table(
        self,
        request_id: str,
        context_id: str,
        command_id: str,
        table_infos: List[TableInfo],
    ) -> TableInfo:
        pass

    # REMOTE TABLES functionality
    def get_remote_tables(self, request_id: str, context_id: str) -> List[str]:
        pass

    def create_remote_table(
        self,
        request_id: str,
        table_name: str,
        table_schema: TableSchema,
        original_db_url: str,
    ) -> TableInfo:
        pass

    # UDFs functionality
    def queue_run_udf(
        self,
        request_id: str,
        context_id: str,
        command_id: str,
        func_name: str,
        positional_args: NodeUDFPosArguments,
        keyword_args: NodeUDFKeyArguments,
        use_smpc: bool = False,
        output_schema: Optional[TableSchema] = None,
    ) -> AsyncResult:
        pass

    def get_queued_udf_result(
        self, async_result: AsyncResult, request_id: str
    ) -> List[NodeUDFDTO]:
        pass

    def get_udfs(self, request_id: str, algorithm_name) -> List[str]:
        pass

    # return the generated monetdb python udf
    def get_run_udf_query(
        self,
        request_id: str,
        context_id: str,
        command_id: str,
        func_name: str,
        positional_args: NodeUDFPosArguments,
    ) -> Tuple[str, str]:
        pass

    # CLEANUP functionality
    def queue_cleanup(self, request_id: str, context_id: str):
        pass

    def wait_queued_cleanup_complete(self, async_result: AsyncResult, request_id: str):
        pass

    # ------------- SMPC functionality ---------------
    def validate_smpc_templates_match(
        self,
        context_id: str,
        table_name: str,
    ):
        pass

    def load_data_to_smpc_client(
        self, context_id: str, table_name: str, jobid: str
    ) -> str:
        pass

    def get_smpc_result(
        self,
        request_id: str,
        jobid: str,
        context_id: str,
        command_id: str,
        command_subid: Optional[str] = "0",
    ) -> TableInfo:
        pass


# --------------------------functions tests----------------------------------
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


# ---------------------------
# class TestDataModelViewsCreator:
#     @pytest.fixture
#     def datasets(self):
#         return ["dataset" + str(i) for i in range(1, 20)]

#     @pytest.fixture
#     def node_mocks(self):
#         context_id = "0"
#         nodes_ids = ["node" + str(i) for i in range(1, 11)]
#         nodes = [
#             create_mock_node(node_id=node_id, context_id=context_id, request_id="0")
#             for node_id in nodes_ids
#         ]
#         return nodes

#     @pytest.fixture
#     def nodes_datasets_mock(self, node_mocks, datasets):
#         nodes_datasets_mock={
#             node_mocks[0]: ["dataset1", "dataset2"],
#             node_mocks[1]: ["dataset2", "dataset3", "dataset6"],
#             node_mocks[2]: ["dataset1", "dataset4", "dataset6"],
#             node_mocks[3]: ["dataset4", "dataset1", "dataset3"],
#             node_mocks[4]: ["dataset2", "dataset1", "dataset6"],
#         }

#         def create_data_model_views_mock(
#             self,
#             command_id: str,
#             data_model: str,
#             datasets: List[str],
#             columns_per_view: List[List[str]],
#             filters: dict = None,
#             dropna: bool = True,
#             check_min_rows: bool = True,
#         )-> str:#-> List[TableInfo]:
#             if set(datasets).issubset(nodes_datasets_mock[self]):
#                 return "a_dummy_view_table"
#             else:
#                 raise InsufficientDataError(f"Node:{self.node_id} has not enough data"
#             )


#         for node,datasets in nodes_datasets_mock.items():
#             node.create_data_model_views=functools.partial(create_data_model_views_mock,node)

#         return nodes_datasets_mock
#         # return {
#         #     node_mocks[0]: ["dataset1", "dataset2"],
#         #     node_mocks[1]: ["dataset2", "dataset3", "dataset6"],
#         #     node_mocks[2]: ["dataset1", "dataset4", "dataset6"],
#         #     node_mocks[3]: ["dataset4", "dataset1", "dataset3"],
#         #     node_mocks[4]: ["dataset2", "dataset1", "dataset6"],
#         # }

#     def test_create_data_model_views(self, nodes_datasets_mock):
#         request_datasets=["dataset2", "dataset6"]

#         init_params= DataModelViewsCreatorInitParams(
#             nodes_datasets=nodes_datasets_mock,
#             data_model="datamodel",
#             datasets=request_datasets,
#             variable_groups=[["var1", "var2"], ["var3"]],
#             var_filters=[],
#             dropna=True,
#             check_min_rows=True,
#             command_id=0,
#         )
#         with unittest.mock.patch('mipengine.controller.controller._data_model_views_to_localnodestables',new=_data_model_views_to_localnodestables_mock):

#             data_model_views_creator=DataModelViewsCreator(init_params)
#             data_model_views_creator.create_data_model_views()
#             # expected
#             expected=[node.node_id for node,datasets in nodes_datasets_mock.items() if set(request_datasets).issubset(datasets)]
#             assert set(expected)==set(data_model_views_creator.data_model_views.to_list())
#         # check that  node.create_data_model_views is called with the relevant params for all nodes
#         # check InsufficientDataError if not enough data on any of the nodes
#         # check it returns the "correct" DataModelViews obj
#         pass


# def _data_model_views_to_localnodestables_mock(
#     views_per_localnode: Dict[LocalNode, List[TableInfo]]
# ) -> List[str]:#List[LocalNodesTable]:
#     # only the node_id portion of the dict is used in the test
#     return [node.node_id for node in views_per_localnode.keys()]

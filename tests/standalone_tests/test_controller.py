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
from mipengine.controller.controller import NodesFederation
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
from mipengine.controller.algorithm_execution_engine import Nodes


from unittest.mock import MagicMock


def create_dummy_node(node_id: str, context_id: str, request_id: str):
    return LocalNode(
        request_id=request_id,
        context_id=context_id,
        node_tasks_handler=DummyNodeAlgorithmTasksHandler(node_id),
        data_model="",
        datasets=[],
    )


@pytest.fixture
def node_mocks():
    context_id = "0"
    nodes_ids = ["node" + str(i) for i in range(1, 11)]
    nodes = [
        LocalNode(
            request_id="0",
            context_id="0",
            node_tasks_handler=DummyNodeAlgorithmTasksHandler(node_id),
            data_model="",
            datasets=[],
        )
        for node_id in nodes_ids
    ]
    return nodes


# @pytest.fixture
# def table_mocks(node_mocks):
#     for node in node_mocks


class TestNodesFederation:
    class NodeInfoMock:
        def __init__(self, node_id: str):
            self.id = node_id
            self.ip = ""
            self.port = ""
            self.db_ip = ""
            self.db_port = ""

        # @property
        # def id(self):
        #     return self._node_id

    class NodeLandscapeAggregatorMock:
        @property
        def nodeids_datasets_mock(self):
            return {
                "node1": ["dataset1", "dataset3", "dataset4"],
                "node2": ["dataset2", "dataset8", "dataset5"],
                "node3": ["dataset9", "dataset6", "dataset7"],
            }

        @property
        def globalnodeid(self):
            return "globalnode"
        
        def get_node_ids_with_any_of_datasets(self, *args, **kwargs):
            return list(self.nodeids_datasets_mock.keys())

        def get_node_info(self, node_id: str):
            return TestNodesFederation.NodeInfoMock(node_id)

        def get_node_specific_datasets(self,node_id: str, data_model: str, wanted_datasets: List[str]):
            return self.nodeids_datasets_mock[node_id]

        def get_global_node(self):
            return TestNodesFederation.NodeInfoMock(self.globalnodeid)
            
    class CommandIdGeneratorMock:
        pass

    class LoggerMock:
        pass

    @pytest.fixture
    def nodes_federation_mock(self):
        return NodesFederation(
            request_id="0",
            context_id="0",
            data_model="",
            datasets=[""],
            var_filters={},
            node_landscape_aggregator=self.NodeLandscapeAggregatorMock(),
            celery_tasks_timeout=0,
            celery_run_udf_task_timeout=0,
            command_id_generator=self.CommandIdGeneratorMock(),
            logger=self.LoggerMock(),
        )

    # def test_get_nodeids_for_requested_datasets(self):
    #     pass
    def test_get_nodeinfo_for_requested_datasets(self):
        pass

    # def test_get_globalnodeinfo(self):
    #     pass
    def test_node_ids(self):
        pass

    def test_create_data_model_views(self):
        pass

    def test_create_nodes(self,nodes_federation_mock):
        nodes_federation=nodes_federation_mock
        created_nodes=nodes_federation._create_nodes()

        assert isinstance(created_nodes,Nodes)

        created_localnodeids=[ localnode.node_id for localnode in created_nodes.local_nodes]
        created_globalnodeid=created_nodes.global_node.node_id
        
        expected_localnodeids=nodes_federation._node_landscape_aggregator.nodeids_datasets_mock.keys()
        assert all(nodeid in created_localnodeids for nodeid in expected_localnodeids)
        
        expected_globalnodeid=nodes_federation._node_landscape_aggregator.globalnodeid
        assert expected_globalnodeid== created_globalnodeid


    def test_get_datasets_of_nodeids(self):
        pass

    def test_create_nodes_tasks_handlers(self):
        pass

    def test_get_nodes_info(self):
        pass


class TestDataModelViews:
    @pytest.fixture
    def views_mocks(self, node_mocks):
        # table naming convention <table_type>_<node_id>_<context_id>_<command_id>_<result_id>
        table_info = TableInfo(
            name=str(TableType.NORMAL).lower()
            + "_"
            + local_node.node_id
            + "_0_"
            + context_id
            + "_0",
            schema_=schema,
            type_=TableType.NORMAL,
        )
        views = [LocalNodesTable(nodes_tables_info={node_mocks[0], table_info})]
        return views
        # breakpoint()
        # schema = TableSchema(
        #     columns=[
        #         ColumnInfo(name="col_1", dtype=DType.INT),
        #         ColumnInfo(name="col_2", dtype=DType.INT),
        #         ColumnInfo(name="col_3", dtype=DType.INT),
        #     ]
        # )

        # nodes_table_info_1 = {}
        # expected_nodes_1 = []
        # for i in range(1, 5):
        #     context_id = "0"
        #     local_node = create_dummy_node(
        #         node_id="node" + str(i),
        #         request_id="0",
        #         context_id=context_id,
        #     )
        #     expected_nodes_1.append(local_node)

        #     table_type = TableType.NORMAL

        #     # table naming convention <table_type>_<node_id>_<context_id>_<command_id>_<result_id>
        #     table_info = TableInfo(
        #         name=str(table_type).lower()
        #         + "_"
        #         + local_node.node_id
        #         + "_0_"
        #         + context_id
        #         + "_0",
        #         schema_=schema,
        #         type_=TableType.NORMAL,
        #     )
        #     nodes_table_info_1[local_node] = table_info

        # local_nodes_table_1 = LocalNodesTable(nodes_tables_info=nodes_table_info_1)

        # nodes_table_info_2 = {}
        # expected_nodes_2 = []
        # for i in range(3, 7):
        #     context_id = "1"
        #     local_node = create_dummy_node(
        #         node_id="node" + str(i),
        #         request_id="0",
        #         context_id=context_id,
        #     )
        #     expected_nodes_2.append(local_node)

        #     table_type = TableType.NORMAL

        #     # table naming convention <table_type>_<node_id>_<context_id>_<command_id>_<result_id>
        #     table_info = TableInfo(
        #         name=str(table_type).lower()
        #         + "_"
        #         + local_node.node_id
        #         + "_0_"
        #         + context_id
        #         + "_0",
        #         schema_=schema,
        #         type_=TableType.NORMAL,
        #     )
        #     nodes_table_info_2[local_node] = table_info

        # local_nodes_table_2 = LocalNodesTable(nodes_tables_info=nodes_table_info_2)

        # return [
        #     (local_nodes_table_1, expected_nodes_1),
        #     (local_nodes_table_2, expected_nodes_2),
        # ]

    def test_get_nodes(self):  # , views_mocks):
        class LocalNodesTable:
            def __init__(self, node_ids: List[str]):
                self.nodes_tables_info = {node_id: None for node_id in node_ids}

        # create a DataModelViews object that contains some LocalNodesTables
        local_node_tables_mock = [
            LocalNodesTable(["node1", "node2", "node3"]),
            LocalNodesTable(["node1", "node8", "node3"]),
            LocalNodesTable(["node1", "node3", "node2"]),
            LocalNodesTable(["node3", "node1", "node8"]),
            LocalNodesTable(["node2", "node1", "node3"]),
        ]
        data_model_views = DataModelViews(local_node_tables_mock)
        result = data_model_views.get_list_of_nodes()

        # get all node_ids from local_node_tables_mock
        tmp = [
            local_node_table.nodes_tables_info.keys()
            for local_node_table in local_node_tables_mock
        ]
        tmp_lists = [list(t) for t in tmp]
        expected_node_ids = set(
            [node_id for sublist in tmp_lists for node_id in sublist]
        )

        assert all(node_id in result for node_id in expected_node_ids)


class TestDataModelViewsCreator:
    # @pytest.fixture
    # def node_mocks(self):
    #     context_id = "0"
    #     nodes_ids = ["node" + str(i) for i in range(5)]
    #     nodes = [
    #         create_mock_node(node_id=node_id, context_id=context_id, request_id="0")
    #         for node_id in nodes_ids
    #     ]
    #     return nodes

    # @pytest.fixture
    # def nodes_datasets_mock(self, node_mocks):
    #     nodes_datasets_mock = {
    #         # node_mocks[0]: ["dataset1", "dataset2", "dataset6"],
    #         # node_mocks[1]: ["dataset2", "dataset3", "dataset6"],
    #         # node_mocks[2]: ["dataset7", "dataset6", "dataset9", "dataset2"],
    #         # node_mocks[3]: ["dataset6", "dataset1", "dataset3", "dataset8", "dataset2"],
    #         # node_mocks[4]: ["dataset2", "dataset6"],
    #         node_mocks[0]: ["dataset1", "dataset3", "dataset4"],
    #         node_mocks[1]: ["dataset2", "dataset8", "dataset5"],
    #         node_mocks[2]: ["dataset9", "dataset6", "dataset7"],
    #     }

    #     has_node_sufficient_data = [
    #         True,
    #         True,
    #         False,
    #         True,
    #         True,
    #         False,
    #         True,
    #         True,
    #         True,
    #         False,
    #     ]
    #     sufficient_data_nodes = dict(zip(node_mocks, has_node_sufficient_data))

    #     def node_create_data_model_views_mock(self, *args, **kwargs) -> str:
    #         requested_datasets = kwargs["datasets"]
    #         subset_of_datasets_contained_in_node = [
    #             requested_dataset
    #             for requested_dataset in requested_datasets
    #             if requested_dataset in nodes_datasets_mock[self]
    #         ]
    #         if subset_of_datasets_contained_in_node:
    #             return [
    #                 f"view_{dataset}"
    #                 for dataset in subset_of_datasets_contained_in_node
    #             ]
    #         # if sufficient_data_nodes[self]:
    #         #     return "a_dummy_view_table"
    #         # else:
    #         #     raise InsufficientDataError(f"Node:{self.node_id} has not enough data")

    #     for node in nodes_datasets_mock.keys():
    #         node.create_data_model_views = functools.partial(
    #             node_create_data_model_views_mock, node
    #         )

    #     return (nodes_datasets_mock, sufficient_data_nodes)

    # check that  node.create_data_model_views is called with the relevant params for all nodes
    # check it returns the "correct" DataModelViews obj
    # @pytest.mark.skip

    @pytest.fixture
    def local_node_mocks(self):
        return [MagicMock(LocalNode) for number_of_nodes in range(10)]

    @pytest.fixture
    def data_model_views_creator_init_params(self, local_node_mocks):
        variable_groups = [["v1," "v2"], ["v3", "v4"]]
        var_filters = []
        dropna = False
        check_min_rows = True
        command_id = 123

        return DataModelViewsCreatorInitParams(
            # local_nodes=[LocalNodeMock()],
            local_nodes=local_node_mocks,
            variable_groups=variable_groups,
            var_filters=var_filters,
            dropna=dropna,
            check_min_rows=check_min_rows,
            command_id=command_id,
        )

    def test_create_data_model_views(
        self, local_node_mocks, data_model_views_creator_init_params
    ):  # , nodes_datasets_mock):
        # class LocalNodeMock:
        #      def create_data_model_views(
        #              self,
        #              command_id: str,
        #              columns_per_view: List[List[str]],
        #              filters: dict = None,
        #              dropna: bool = True,
        #              check_min_rows: bool = True,
        #      ):
        #          return ""

        # local_node_mocks = [MagicMock(LocalNode) for number_of_nodes in range(10)]
        # variable_groups = [["v1," "v2"], ["v3", "v4"]]
        # var_filters = []
        # dropna = False
        # check_min_rows = True
        # command_id = 123

        # init_params = DataModelViewsCreatorInitParams(
        #     # local_nodes=[LocalNodeMock()],
        #     local_nodes=local_node_mocks,
        #     variable_groups=variable_groups,
        #     var_filters=var_filters,
        #     dropna=dropna,
        #     check_min_rows=check_min_rows,
        #     command_id=command_id,
        # )
        data_model_views_creator = DataModelViewsCreator(
            data_model_views_creator_init_params
        )
        # assert that create_data_model_views was called for all self._local_nodes with the args
        data_model_views_creator.create_data_model_views()

        for node in local_node_mocks:
            node.create_data_model_views.assert_called_once_with(
                columns_per_view=data_model_views_creator_init_params.variable_groups,
                filters=data_model_views_creator_init_params.var_filters,
                dropna=data_model_views_creator_init_params.dropna,
                check_min_rows=data_model_views_creator_init_params.check_min_rows,
                command_id=data_model_views_creator_init_params.command_id,
            )

        assert isinstance(data_model_views_creator.data_model_views, DataModelViews)

        # nodes_datasets_mock, sufficient_data_nodes = nodes_datasets_mock
        # request_datasets = ["dataset1", "dataset2", "dataset4", "dataset7", "dataset8"]

        # init_params = DataModelViewsCreatorInitParams(
        #     nodes_datasets=nodes_datasets_mock,
        #     data_model="datamodel",
        #     datasets=request_datasets,
        #     variable_groups=[],
        #     var_filters=[],
        #     dropna=True,
        #     check_min_rows=True,
        #     command_id=0,
        # )
        # with unittest.mock.patch(
        #     "mipengine.controller.controller._data_model_views_to_localnodestables",
        #     new=_data_model_views_to_localnodestables_mock,
        # ):
        #     data_model_views_creator = DataModelViewsCreator(init_params)
        #     data_model_views_creator.create_data_model_views()

        #     expected = [
        #         node.node_id
        #         for node, has_sufficient_data in sufficient_data_nodes.items()
        #         if has_sufficient_data
        #     ]

        #     assert set(expected) == set(
        #         data_model_views_creator.data_model_views.to_list()
        #     )

    def test_create_data_model_views_insufficient_data_error(
        self, data_model_views_creator_init_params
    ):

        local_node_mocks = [MagicMock(LocalNode) for number_of_nodes in range(10)]
        for node_mock in local_node_mocks:
            node_mock.node_id = "some_id.."
            node_mock.create_data_model_views.side_effect = InsufficientDataError("")
            # node_mock.create_data_model_views.side_effect = (
            #     lambda *args: InsufficientDataError("")
            # )

        data_model_views_creator_init_params = DataModelViewsCreatorInitParams(
            local_nodes=local_node_mocks,
            variable_groups=data_model_views_creator_init_params.variable_groups,
            var_filters=data_model_views_creator_init_params.var_filters,
            dropna=data_model_views_creator_init_params.dropna,
            check_min_rows=data_model_views_creator_init_params.check_min_rows,
            command_id=data_model_views_creator_init_params.command_id,
        )
        # breakpoint()
        data_model_views_creator = DataModelViewsCreator(
            data_model_views_creator_init_params
        )
        # assert that create_data_model_views was called for all self._local_nodes with the args
        with pytest.raises(InsufficientDataError):
            data_model_views_creator.create_data_model_views()

    # check InsufficientDataError if not enough data on any of the nodes
    @pytest.mark.skip
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


# def _data_model_views_to_localnodestables_mock(
#     views_per_localnode: Dict[LocalNode, List[TableInfo]]
# ) -> List[str]:  # List[LocalNodesTable]:
#     # only the node_id portion of the dict is used in the test
#     return [node.node_id for node in views_per_localnode.keys()]


class AsyncResult:
    pass


class DummyNodeAlgorithmTasksHandler(INodeAlgorithmTasksHandler):
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

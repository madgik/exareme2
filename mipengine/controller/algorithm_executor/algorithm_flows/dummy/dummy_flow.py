from AlgorithmExecutor import AlgorithmFlow_abstract
from data_classes import TableInfo, TableData,TableView, ColumnInfo

import pdb


class AlgorithmFlow(AlgorithmFlow_abstract):

    def run(self):
        print("\n(dummy_flow.py::AlgorithmFlow::run) just in")
        runtime_interface = self.runtime_interface

        # data = runtime_interface.get_table_data_from_local0(runtime_interface.local_nodes[0].initial_view_table)
        # print(f"data->{data}")

        # # NODE insides..
        # global_node_initial_view_table = runtime_interface.global_node.initial_view_table
        # print(f"global_node_initial_view_table->{global_node_initial_view_table}")

        # node1_initial_view_table = runtime_interface.local_nodes[0].initial_view_table
        # print(f"node1_initial_view_table->{node1_initial_view_table.full_table_name}")

        # node2_initial_view_table_name = runtime_interface.local_nodes[1].initial_view_table
        # print(f"node2_initial_view_name->{node2_initial_view_table_name}")

        # node1_initial_view_table_schema = runtime_interface.local_nodes[1].get_table_schema(runtime_interface.local_nodes[1].initial_view_table)
        # print(f"node1_base_view_schema->{node1_initial_view_table_schema}   type->{type(node1_initial_view_table_schema)}")

        # node1_initial_view_data = runtime_interface.local_nodes[0].get_table_data(runtime_interface.local_nodes[0].initial_view_table)
        # print(f"node1_initial_view_data->{node1_initial_view_data}")

        # node2_initial_view_data = runtime_interface.local_nodes[1].get_table_data(runtime_interface.local_nodes[1].initial_view_table)
        # print(f"\nnode2_initial_view_data->{node2_initial_view_data}")

        return "ok"

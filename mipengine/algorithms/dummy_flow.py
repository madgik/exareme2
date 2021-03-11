import time

def run(runtime_interface):
    print("\n(dummy_flow.py::AlgorithmFlow::run) just in")

    runtime_interface.initial_view_tables["x"]

    node = runtime_interface.local_nodes[0]
    table = runtime_interface.local_nodes[0].initial_view_tables["x"]
    data = runtime_interface.get_table_data_from_local(node, table)
    # print(f"data->{data}\n")

    #DEBUG
    # time.sleep(15)

    # node = runtime_interface.local_nodes[1]
    # table = runtime_interface.local_nodes[1].initial_view_tables["y"]
    # data = runtime_interface.get_table_data_from_local(node, table)
    # print(f"data->{data}\n")

    # # # NODE insides..
    # global_node_initial_view_tables = runtime_interface.global_node.initial_view_tables
    # print(f"global_node_initial_view_tables->{global_node_initial_view_tables}")

    # node0_initial_view_table_x = runtime_interface.local_nodes[0].initial_view_tables["x"]
    # print(f"node0_initial_view_table->{node0_initial_view_table_x.full_table_name}")

    # node1_initial_view_table_x = runtime_interface.local_nodes[1].initial_view_tables["x"]
    # print(f"node1_initial_view_table->{node1_initial_view_table_x.full_table_name}")

    # node = runtime_interface.local_nodes[1]
    # table = runtime_interface.local_nodes[1].initial_view_tables["x"]
    # node1_initial_view_table_schema_x = node.get_table_schema(table)
    # print(f"node1_base_view_schema_x->{node1_initial_view_table_schema_x}   type->{type(node1_initial_view_table_schema_x)}")

    # table = runtime_interface.local_nodes[1].initial_view_tables["y"]
    # node1_initial_view_table_schema_y = node.get_table_schema(table)
    # print(f"node1_base_view_schema_y->{node1_initial_view_table_schema_y}   type->{type(node1_initial_view_table_schema_y)}")


    # return "ok"
    return data

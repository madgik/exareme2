#import run_algorithm #yeah... fix those names...
from run_algorithm import AlgorithmFlow_abstract
from pprint import pprint
import pdb

class AlgorithmFlow(AlgorithmFlow_abstract):
    
    async def run(self):
        runtime_interface=self.runtime_interface

        #TODO:can the result schema be inferred instead of declared here
        udf_result_schema="return1 FLOAT,return2 FLOAT"

        result_dict=await runtime_interface.execute_udf_on_all_local_nodes("local_calc",udf_result_schema,self.data_table_name)
        #result_dict is {node_object:table_name}
        print(f"\n(dummy_flow.py::AlgorithmFlow::run) result_dict->")
        pprint(result_dict)

        #..just to print the result data..
        for node in result_dict.keys():
            print(f"\n(dummy_flow.py::AlgorithmFlow::run) \nNODE-> {node._db_name} \nresult_dict[{node}]->{result_dict[node]}")
            result=await node.execute_select(result_dict[node])
            print(f"(dummy_flow.py::AlgorithmFlow::run) result-> {result}\n")

        #make the result tables "visible" to the global node
        merged_table_name=await runtime_interface.create_remote_locals_to_global(result_dict)

        result_table_name=await runtime_interface.execute_udf_on_global_node("global_calc",udf_result_schema,merged_table_name)

        #make the result table "visible" to the local nodes
        await runtime_interface.create_remote_global_to_locals(result_table_name)

        #Since this is a dummy, just return the remote table data from one of the local nodes
        node=runtime_interface.nodes["local"][0]
        print(f"\n(dummy_flow.py::AlgorithmFlow::run) returning REMOTE table:{result_table_name} from LOCAL node:{node._db_name}")
        result=await node.execute_select(result_table_name)

        return result

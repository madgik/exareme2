import datetime
import importlib
import inspect
import json
import parser
import random
import re
import asyncio

from abc import ABC

#DEBUG
import pdb

async def run(algorithm_json,data_json,db_objects):
    print(f"(run_algorithm.py::run) \nalgorithm_json->{algorithm_json} \ndata_json->{data_json}\n")
    
    algorithm_name=algorithm_json["files"]["algorithmFolder"]
    algorithm_folder=algorithm_json["files"]["algorithmFolder"]
    algorithm_flow_file=algorithm_json["files"]["algorithmFlowFile"]
    algorithm_udfs_file=algorithm_json["files"]["algorithmUDFsFile"]

    algorithm_flow_module_handle = importlib.import_module(f"algorithms.{algorithm_folder}.{algorithm_flow_file}")
    algorithm_udfs_module_handle=importlib.import_module(f"algorithms.{algorithm_folder}.{algorithm_udfs_file}")

    print(f"\n(run_algorithm.py::run) algorithm_folder-> {algorithm_folder}")
    print(f"(run_algorithm.py::run) algorithm_flow_file-> {algorithm_flow_file}")
    print(f"(run_algorithm.py::run) algorithm_flow_module-> {algorithm_flow_module_handle}")
    print(f"(run_algorithm.py::run) algorithm_udfs_module-> {algorithm_udfs_module_handle}")

    algorithm_info={"name":algorithm_name,"flow_module_handle":algorithm_flow_module_handle,"udfs_module_handle":algorithm_udfs_module_handle}

    runtime_interface=RuntimeInterface(db_objects,algorithm_info)

    initial_data_table_name=data_json["table"]
    attributes=data_json["attributes"]
    filters=data_json["filters"]
    data_view_name=await runtime_interface.create_data_views_on_all_local_nodes(initial_data_table_name,attributes,filters)

    algorithm_parameters=algorithm_json["parameters"]
    algorithm_flow=algorithm_flow_module_handle.AlgorithmFlow(runtime_interface,data_view_name,algorithm_parameters)

    result=await algorithm_flow.run()

    return result


class AlgorithmFlow_abstract(ABC):
    def __init__(self,runtime_interface,data_table_name,algorithm_parameters):
        self.runtime_interface=runtime_interface
        self.data_table_name=data_table_name
        self.algorithm_parameters=algorithm_parameters

class RuntimeInterface:
    
    def __init__(self,db_objects,algorithm_info):
        self._nodes={}

        global_node=self.Node(url=db_objects["global"]["dbname"],connection_obj=db_objects["global"]["async_con"],db_name=db_objects["global"]["host"],alias_name="global_node")
        self._nodes["global"]=global_node

        self._nodes["local"]=[]
        for db_object in db_objects["local"]:
            local_node=self.Node(url=db_object["dbname"],connection_obj=db_object["async_con"],db_name=db_object["host"],alias_name=db_object["dbname"])
            self._nodes["local"].append(local_node)

        self.algorithm_info=algorithm_info

    async def create_data_views_on_all_local_nodes(self, table_name,attributes=[],filters=[]):
        view_name=f"{table_name}_{get_a_uniqueid()}"
        create_view_calls = [ local_node.create_data_view(view_name,table_name,attributes,filters) for local_node in self.nodes["local"] ]
        await asyncio.gather(*create_view_calls)

        return view_name

    async def execute_udf_on_all_local_nodes(self,udf_name,udf_result_schema,input_table_name):
        print(f"\n(run_algorithm.py::RuntimeInterface::execute_udf_on_all_local_nodes) Will execute '{udf_name}' on ALL LOCAL NODES")
        udfs_module_handle=self.algorithm_info["udfs_module_handle"]
        udf_result_tables = { local_node : await local_node.execute_udf(udfs_module_handle,udf_name,udf_result_schema,input_table_name) for local_node in self.nodes["local"] }

        return udf_result_tables

    async def execute_udf_on_global_node(self,udf_name,udf_result_schema,input_table):
        print(f"\n(run_algorithm.py::RuntimeInterface::execute_udf_on_global_node) Will execute '{udf_name}' on the GLOBAL NODE")
        result_table = await self.nodes["global"].execute_udf(udf_name,udf_name,udf_result_schema,input_table) 
        return result_table

    async def create_remote_locals_to_global(self,node_table_name_tuple):
        #TODO: table table_name must exist on all local nodes. Check it..
        schema=""
        for node,table_name in node_table_name_tuple.items():
            print(f"(run_algorithm.py::RuntimeInterface::create_remote_locals_to_global) Creating remote LOCAL table:{table_name} from {node._db_name} on GLOBAL node")
            schema=await self.get_table_schema(table_name,node)
            await self.nodes["global"].create_remote_table(table_name,schema,node)

        #TODO define and establish some kind of convention concerning the names of the tables ex. data_123455_blabla_blabla date is the view, 123456 is the current algorithm orchestration identifier
        merged_table_name=next(iter(node_table_name_tuple.values())).split("_") #TODO:yeah I know..
        merged_table_name=f"{merged_table_name[0]}_{merged_table_name[1]}"
        print(f"(run_algorithm.py::RuntimeInterface::create_remote_locals_to_global) MERGING on GLOBAL node, the tables as ->{merged_table_name}")
        await self.nodes["global"].merge_tables(merged_table_name,schema,list(node_table_name_tuple.values()))

        return merged_table_name

    async def create_remote_global_to_locals(self,table):
        for node in self.nodes["local"]:
            print(f"\n(run_algorithm.py::RuntimeInterface::create_remote_global_to_locals) Creating remote GLOBAL table:{table} on LOCAL node:{node._db_name}")
            global_node=self.nodes["global"]
            schema=await self.get_table_schema(table,global_node)
            await node.create_remote_table(table,schema,global_node)

    async def get_table_schema(self,table,node):
        return await node.get_table_schema(table)

    async def get_table_columns(self, table_name,node):
        return await node.get_table_columns(table_name)

    async def get_number_of_columns(self,table_name,node):
        query=f"SELECT COUNT(*) FROM columns INNER JOIN tables ON columns.table_id=tables.id AND tables.name='{table_name}'"
        response=await node.execute_query(query)
        return response[0][0]

    async def get_data_view_columns(self):
        #TODO:check all nodes have the correct view??
        return await self.nodes["local"][0].get_table_columns(self.data_table_name)

    @property
    def nodes(self):
        return self._nodes

    class Node:
        def __init__(self,url,connection_obj,db_name,alias_name):
            self._url=url
            self._connection_obj=connection_obj
            self._db_name=db_name
            self._alias_name=alias_name

#        @property
#        def db_alias_name(self):
#            return self._alias_name

        async def get_table_schema(self,table):
            query=f"SELECT columns.name,columns.type FROM columns INNER JOIN tables ON columns.table_id=tables.id AND tables.name='{table}'"
            response=await self.execute_query(query)
            schema=[f"{name} {type}" for name,type in response]
            schema=",".join(schema)
            return schema

        async def execute_udf(self,udfs_module_handle,udf_name,udf_result_schema,input_table_name):
            print(f"\n(run_algorithm.py::Node::execute_udf) NODE:{self._db_name} \nExecuting udf-> {udf_name} with udf_result_schema->({udf_result_schema}), input_table_name->{input_table_name}")

            mod=udfs_module_handle #handle to the module containing the udfs code

            udf_input_schema=await self.get_table_schema(input_table_name)

            #Generating python UDF...
            python_udf=await self.python_to_pythonUDF(udfs_module_handle,udf_name,input_table_name,udf_result_schema)

            udf_function_name=python_udf["udf_name"]
            udf_function_code=python_udf["code"]

            await self.execute_query(udf_function_code) #Creates the udf on the node's DB
            
            #There needs to be a table that will hold the results of the UDF execution
            #TODO: Make some kind of convention on the naming of the generated tables, to facilitate later debugging
            result_table_name=f"{input_table_name}_{self._db_name}_{udf_name}_{get_a_uniqueid()}"
            await self.create_table(result_table_name,udf_result_schema)
                
            #call the udf
            query=f"INSERT INTO {result_table_name} SELECT * FROM {udf_function_name}((SELECT * FROM {input_table_name}));"
            await self.execute_query(query)

            return result_table_name

        #THIS IS A PLACEHOLDER FOR DEMONSTRATION
        #The actual transformation from pure python to python UDF syntax is missing here...
        async def python_to_pythonUDF(self,udfs_module_handle,udf_name,input_table_name,result_schema):
            print(f"(run_algorithm.py::Node::python_to_pythonUDF) NODE: {self._db_name} Generating python UDF ...")
            table_columns=await self.get_table_columns(input_table_name)
            input_table_schema=await self.get_table_schema(input_table_name)

            unique_udf_name=f"{udf_name}_{get_a_uniqueid()}"

            if udf_name=="local_calc":
                #as this is just a placeholder, the function body would normally be parsed from the pure python code
                function_body=  f'to_return={{}} \n'\
                                f'to_return["return1"]=input_par1 \n'\
                                f'to_return["return2"]=input_par2 \n'\
                                f'return to_return '

            elif udf_name=="global_calc":
                #as this is just a placeholder, the function body would normally be parsed from the pure python code
                function_body=f'to_return={{}} \n'\
                            f'to_return["return1"]=input_par1 \n'\
                            f'to_return["return2"]=input_par2 \n'\
                            f'return to_return '

            
            
            code=f'CREATE OR REPLACE FUNCTION {unique_udf_name}({input_table_schema}) \n' \
                f'RETURNS TABLE({result_schema}) \n'\
                f'LANGUAGE PYTHON {{\n'\
                f'input_par1={list(table_columns.keys())[0]}; input_par2={list(table_columns.keys())[1]}\n'\
                f'{function_body} \n}}'
            
            to_return={"udf_name":unique_udf_name,"code":code}
            return to_return

        async def create_table(self,table_name: str, table_schema: str):
            query = f"CREATE TABLE {table_name} ({table_schema});"
            await self.execute_query(query)

        async def insert_values_to_table(self,table_name,values):
            def list_shape(a):
                if not type(a) == list:
                    return []
                return [len(a)] + list_shape(a[0])
            
            values_list_dimensions=len(list_shape(values))
            if values_list_dimensions==1:
                values=f'( {",".join(map(str,values))} ) '
            elif values_list_shape==2:
                values=",".join(map(str,values)).replace("[","(").replace("]",")")
            else:
                raise Exception(f"(run_algorithm.py::Node::create_table) values_list_dimensions={values_list_dimensions} BUT should be MAX 2")
            
            query= f"INSERT INTO {table_name} VALUES {values}"
            await self.execute_query(query)

        async def create_remote_table(self, table_name,table_schema,native_node):
            query = f"CREATE REMOTE TABLE {table_name} ({table_schema}) on 'mapi:{native_node._url}';"
            await self.execute_query(query)

        async def merge_tables(self,merged_table_name,merged_table_schema,tables_to_be_merged):
            #TODO:should not be just dropped if it exists. If it exists it could mean there was a name colision and should be signaled, exception or somethin..
#           query=f"DROP TABLE IF EXISTS {merge_table_name}"

            query=f"CREATE MERGE TABLE {merged_table_name} ({merged_table_schema})"
            await self._connection_obj.cursor().execute(query)

            for table in tables_to_be_merged:
                query=f"ALTER TABLE {merged_table_name} ADD TABLE {table}"
                await self._connection_obj.cursor().execute(query)

        async def create_data_view(self, view_name,data_table_name,attributes=[],filters=[]):
                print(f"(run_algorithm.py::Node::create_data_view) Creating view:{view_name} on node: {self._url}")

#                for formula in params["filters"]:
#                    for attribute in formula:
#                        if attribute[0] not in attributes:
#                            attributes.append(attribute[0])
#                print(f"(run_algorithm.py::Node::create_data_view) data_table->{data_table}")
#                await self._connection_obj.check_for_params(data_table, attributes)#raises exception in case of failure, change it to return bool??
#                    
#                #TODO:hide this abomination
#                filterpart = " "
#                vals = []
#                for j, formula in enumerate(params["filters"]):
#                    andpart = " "
#                    for i, filt in enumerate(formula):
#                        if filt[1] not in [">", "<", "<>", ">=", "<=", "="]:
#                            raise Exception("Operator " + filt[1] + " not valid")
#                        andpart += filt[0] + filt[1] + "%s"
#                        vals.append(filt[2])
#                        if i < len(formula) - 1:
#                            andpart += " and "
#                    if andpart != " ":
#                        filterpart += "(" + andpart + ")"
#                    if j < len(params["filters"]) - 1:
#                        filterpart += " or "

                #TODO: filterpart is ignored...
                filterpart=" "

                if attributes:
                    attributes=",".join(attributes)
                else:
                    attributes="*"

                query_partial = f"CREATE VIEW {view_name} AS SELECT {attributes} FROM {data_table_name}"

                if filterpart==" ":
                    query=f"{query_partial} ;"
                    await self.execute_query(query)
                else:
                    query=f"{query_partial} WHERE {filterpart} ;"
                    await self.execute_query(query,vals)

        async def get_table_columns(self, table_name):
            query=f"SELECT columns.name,columns.type FROM columns INNER JOIN tables ON columns.table_id=tables.id AND tables.name='{table_name}'"
            result=await self.execute_query(query)
            result_dict={}
            for item in result:
                result_dict[item[0]]=item[1]
            return result_dict


        async def execute_select(self,table_name,attributes="*"):
            query=f"SELECT {attributes} FROM {table_name}"
            return await self.execute_query(query)

        async def execute_query(self,query,vals=None):
            cursor= self._connection_obj.cursor()
            if vals:
                exec_res=await cursor.execute(query, vals)
            else:
                exec_res=await cursor.execute(query)

            try:
                return cursor.fetchall()
            except:
                #TODO: if there is nothing to fecth fetchall() raises exception, how should this be dealed with?
                #print(f"(run_algorithm.py::Node::execute_query) fetchall raised exception... ignored")
                pass

def get_a_uniqueid():
    return "{}".format(datetime.datetime.now().microsecond + (random.randrange(1, 100 + 1) * 100000))

def get_package(algorithm):
    mpackage = "algorithms"
    importlib.import_module(mpackage)
    algo = importlib.import_module("." + algorithm, mpackage)
    return algo



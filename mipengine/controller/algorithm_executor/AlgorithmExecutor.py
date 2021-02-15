#import datetime
#import inspect
#import json
#import parser
#import random
#import re
#import asyncio

from celery import Celery
from pprint import pprint
from abc import ABC

import importlib

import pdb

#Node
from collections import deque
import threading

#from typing import Final
from data_classes import TableInfo, TableData,TableView, ColumnInfo, UDFInfo, Parameter
from typing import List

TASK_TIMEOUT=10

#----------------------------------------
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import Optional

from dataclasses_json import dataclass_json

#------------------------------------------
@dataclass_json
@dataclass
class AlgorithmRequestDTO:
    inputdata: "AlgorithmInputData"
    parameters: Optional[Dict[str,] = None
    crossvalidation: Optional[Dict[str, Any]] = None

@dataclass_json
@dataclass
class AlgorithmInputData:
    pathology:str
    datasets:[str]
    filters:[str]
    x:[str]
    y:[str]
#---------------------------------------
@dataclass_json
@dataclass
class Node:
    nodeId: str
    rabbitmqURL: str
    monetdbURL: str
    data: Optional[["Pathology"]]=None #global node will not have pathologies

@dataclass_json
@dataclass
class Pathology:
    name: str
    datasets: List[str]


@dataclass_json
@dataclass
class NodeCatalogue(metaclass=Singleton):
#---------------------------------------
class AlgorithmExecutor:
    def __init__(self,algorithm_request:"AlgorithmRequestDTO"):

        print(f"(AlgorithmExecutor::__init__) just in")

        self.context_id=get_a_uniqueid() #TODO this should be passed as a param??


        global_node=NodeDataCatalog.get_global_node()
        local_nodes=NodeCatalogue.get_nodes_with_datasets(algorithm_request.datasets)
        
        #instantiate the Node objects
        self.global_node=self.Node(rabbitmq_url=global_node.rabbitmqURL, monetdb_url=global_node.monetdbURL,context_id=context_id)

        base_view_table_params={"datasets"=algorithm_request.datasets, "columns":(algorithm_request.x+algorithm_request.y)}
        self.local_nodes=[]
        for local_node in local_nodes:
            self.local_nodes.append(self.Node(rabbitmq_url=local_node.rabbitmqURL, monetdb_url=local_node.monetdbURL,,base_view_table_params=base_view_table_params,context_id=context_id))

    
        #TODO: create a view on each node and store it on a dict
        task_per_nodes={}
        for node in self.local_nodes:
            base_view_table_params={"datasets"=algorithm_request.datasets, "columns":(algorithm_request.x+algorithm_request.y)}
            task=node.create_view(datasets=algorithm_request.datasets, columns=(algorithm_request.x+algorithm_request.y) )
            task=node.create_view(datasets=algorithm_request.datasets, columns=(algorithm_request.x+algorithm_request.y) )
            task_per_nodes[node]=task

        nodes_and_base_view_tables={}#{Node,view_table_name}
        for (node,task) in task_per_nodes.items:
            nodes_and_base_view_tables[node]=task.get()

        self.algorithm_params=algorithm_params

        execution_interface=self.AlgorithmExecutionInterface(global_node=self.global_node, nodes_and_base_view_tables=nodes_and_base_view_tables, algorithm_params=AlgorithmRequestDTO.parameters)

        
        #import the algorithm flow module
        algorithm_folder=self.algorithm_params["algorithmFolder"]
        algorithm_flow_file=self.algorithm_params["algorithmFlowFile"]
        algorithm_flow_module= importlib.import_module(f"algorithm_flows.{algorithm_folder}.{algorithm_flow_file}")
        print(f"(AlgorithmExecutor::__init__) algorithm_flow_module->{algorithm_flow_module}")

        
        algorithm_flow_module.AlgorithmFlow(execution_interface).run()


        class Node:
            def __init__(self,rabbitmq_url,monetdb_url,base_view_table_params,context_id):
                
                self.__celery_obj=Celery(broker=celery_queue_url,backend='rpc://')
                self.__monetdb_url=monetdb_url
                self.__base_view_table_task=self._create_base_view_table(base_view_table_params)
                self.base_view_table=None
                

            @property
            def base_view_table(self):
                #if self.base_view_table==None:
                return self.__base_view_table_task.get()   
                 
            def create_base_views(self,datasets, columns)
                #the base_view_table_name will be the view where the algorithm will do its thing on..
                create_view(datasets=datasets,columns=columns)
                
            #TABLES
            def get_table_info(self,table_names: List[str] = None):
                task_signature = self.__celery_obj.signature('worker.tasks.tables.get_tables_info')
                return task_signature.delay([table_names])
                
            def create_table(self,schema: List[ColumnInfo], execution_id: str):                
                task_signature = self.__celery_obj.signature('worker.tasks.tables.create_table')
                return task_signature.delay(schema,execution_id)

            def get_table_data(table_name: str) -> TableData:                
                task_signature = self.__celery_obj.signature('worker.tasks.tables.get_table_data')
                return task_signature.delay(schema,table_name)

            def delete_table(table_name: str):
                task_signature = self.__celery_obj.signature('worker.tasks.tables.delete_table')
                return task_signature.delay(schema,table_name)

            #VIEWS
            def get_views() -> List[TableInfo]:
                task_signature = self.__celery_obj.signature('worker.tasks.views.')
                return task_signature.delay(schema,table_name)
            def create_view(view: TableView) -> TableInfo:
                pass
            def get_view(view_name: str) -> TableData:
                pass
            def delete_view(view_name: str):
                pass
            #MERGE TABLES
            def get_merge_tables() -> List[TableInfo]:
                pass
            def create_merge_table(schema: str):
                pass
            def get_merge_table(merge_table_name: str) -> TableData:
                pass
            def update_merge_table(merge_table_name: str):
                pass
            #REMOTE TABLES
            def get_remote_tables() -> List[TableInfo]:
                pass
            def create_remote_table(table_name: str):
                pass
            def get_remote_table(remote_table_name: str) -> TableData:
                pass
            #UDFs
            def get_udfs() -> List[UDFInfo]:
                pass
            def run_udf(udf_name: str, input: List[Parameter]) -> TableInfo:
                pass
            def get_udf(udf_name: str) -> UDFInfo:
                pass


    class AlgorithmExecutionInterface:
        
        def __init__(self,global_node, nodes_and_base_view_tables, algorithm_params):
            self.global_node=global_node
            self.nodes_and_base_view_tables=nodes_and_base_view_tables

        #UDFs
        def run_udf_on_locals_merge_on_global(self, udf_name: str, input: List[Parameter])->TableInfo:
            #TREAT ALL NODES AS BEEING ONE NODE???
            #<tableType>_<contextIdentifier>_<nodeIdentifier>_<UUID> -> <tableType>_<tableId>_<contextIdentifier>_<nodeIdentifier>
            #expose only <tableType>_<tableId>_<contextIdentifier> to the algorithm developper?

            #queue exec_udf task on all local nodes
            #wait for all nodes to complete the tasks execution
            #one new table per local node was generated 
            #queue create_remote_table on global for each of the generated tables
            #create merge table on global node to merge the remote tables
            pass
        def run_udf_on_global_remotes_on_locals(self, udf_name: str, input: List[Parameter]):#-> (node,table_name)
            #queue exec_udf on the global node
            #wait for it to complete
            #a new table was generated on global node
            #queue create_remote_table on each of the local nodes for the ganerated table
            pass


        #TABLES
        def get_table_data_from_global(self, table_name: str)->TableData:
            pass

                


class AlgorithmFlow_abstract(ABC):
    def __init__(self,execution_interface):

        self.runtime_interface=execution_interface

def get_a_uniqueid():
    return "{}".format(datetime.datetime.now().microsecond + (random.randrange(1, 100 + 1) * 100000))

def get_package(algorithm):
    mpackage = "algorithms"
    importlib.import_module(mpackage)
    algo = importlib.import_module("." + algorithm, mpackage)
    return algo


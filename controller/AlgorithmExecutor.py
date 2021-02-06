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

class AlgorithmExecutor:
    def __init__(self,algorithm_params,node_params):
        self.context_id=1234 #TODO this should be passed as a param??
        
        print(f"(AlgorithmExecutor::__init__) just in")
        self.algorithm_params=algorithm_params
        self.node_params=node_params

        #import the algorithm flow module
        algorithm_folder=self.algorithm_params["algorithmFolder"]
        algorithm_flow_file=self.algorithm_params["algorithmFlowFile"]
        algorithm_flow_module= importlib.import_module(f"algorithm_flows.{algorithm_folder}.{algorithm_flow_file}")
        print(f"(AlgorithmExecutor::__init__) algorithm_flow_module->{algorithm_flow_module}")

        execution_interface=self.ExecutionInterface(node_params)
        
        algorithm_flow_module.AlgorithmFlow(execution_interface).run()

    class ExecutionInterface:
        
        def __init__(self,node_params):

            #TODO: hide actual Node objects, expose some kind of alias?
            #instantiate the Node objects
            self.global_node=self.Node(node_params["globalNode"]["url"])
            self.local_nodes=[]
            for local_node_params in node_params["localNodes"]:
                self.local_nodes.append(self.Node(local_node_params["url"],local_node_params["viewName"]))

        #UDFs
        def run_udf_on_locals_merge_on_global(self, udf_name: str, input: List[Parameter])->TableInfo:
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

                

        class Node:
            def __init__(self,celery_queue_url,base_view_table_name=None):
                
                self.__celery_obj=Celery(broker=celery_queue_url,backend='rpc://')

                #the base_view_table_name will be the view where the algorithm will do its thing on..
                self.__base_view_table_name=base_view_table_name
                
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


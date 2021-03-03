from __future__ import annotations
from typing import Dict, List, Any, Optional

import json
import datetime
import random
from re import search

from abc import ABC

import importlib

from celery import Celery


#  TODO: change those to the actual ones...
from node_catalog import NodeCatalog
from AlgorithmRequestDTO import AlgorithmRequestDTO
from data_classes import ColumnInfo, TableSchema, TableInfo 
from data_classes import TableView, TableData, Parameter, UDFInfo, UDFInput

import pdb
import pprint

TASK_TIMEOUT = 2


class AlgorithmExecutor:
    def __init__(self, algorithm_request: AlgorithmRequestDTO):

        print("(AlgorithmExecutor::__init__) just in")

        self.context_id = get_a_uniqueid()  # TODO this should be passed as a param?? daasdddasdsdasdasddx

        node_catalog = NodeCatalog()
        global_node = node_catalog.get_global_node()
        local_nodes = node_catalog.get_nodes_with_datasets(algorithm_request.datasets)

        # TODO: As is now, if multiple datasets are passed only the nodes of the 1st dataset will be processed
        # TODO: node_catalog returns List[List[Node]] this has to be fixed there
        local_nodes=[node for node in local_nodes[0]]

        # DEBUG
        print(f"(AlgorithmExecutor::__init__) local_nodes->{[node.nodeId for node in local_nodes]}\n\n")
       
        # instantiate the Node objects
        self.global_node = self.Node(node_id=global_node.nodeId,
                                     rabbitmq_url=global_node.rabbitmqURL,
                                     monetdb_url=f"{global_node.monetdbHostname}:{global_node.monetdbPort}",
                                     context_id=self.context_id)

        base_view_table_params = {"commandId": get_a_uniqueid(),
                                  "pathology": algorithm_request.pathology,
                                  "datasets": algorithm_request.datasets,
                                  "columns": (algorithm_request.x + algorithm_request.y),
                                  "filters": algorithm_request.filter}

        #dEBUG
        print(f"(AlgorithmExecutor::__init__) base_view_table_params['commandId']->{base_view_table_params['commandId']}")
        self.local_nodes = []
        for local_node in local_nodes:
            # pdb.set_trace()
            #local_node = local_node[0]  # facepalm
            self.local_nodes.append(self.Node(node_id=local_node.nodeId,
                                              rabbitmq_url=local_node.rabbitmqURL,
                                              monetdb_url=f"{local_node.monetdbHostname}:{global_node.monetdbPort}",
                                              base_view_table_params=base_view_table_params,
                                              context_id=self.context_id))

        execution_interface = self.AlgorithmExecutionInterface(global_node=self.global_node,
                                                               local_nodes=self.local_nodes,
                                                               algorithm_params=algorithm_request)

        # import the algorithm flow module
        algorithm_folder = algorithm_request.algorithm_name  # self.algorithm_params["algorithmFolder"]
        algorithm_flow_file = algorithm_request.algorithm_name  # ["algorithmFlowFile"]
        print(f"will import-> algorithm_flows.{algorithm_folder}.{algorithm_flow_file}")
        algorithm_flow_module = importlib.import_module(f"algorithm_flows.{algorithm_folder}.{algorithm_flow_file}_flow")
        print(f"(AlgorithmExecutor::__init__) algorithm_flow_module->{algorithm_flow_module}")

        algorithm_flow_module.AlgorithmFlow(execution_interface).run()

    class Node:
        def __init__(self, node_id, rabbitmq_url, monetdb_url, context_id, base_view_table_params=None):

            self.node_id = node_id

            # TODO: user, pass, vhost how these should be set??
            user = "user"
            password = "password"
            vhost = "user_vhost"
            broker = f"amqp://{user}:{password}@{rabbitmq_url}/{vhost}"
            self.__celery_obj = Celery(broker=broker, backend='rpc://')
            print(f"(Node::__init__) rabbitmq_url->{rabbitmq_url}")

            self.monetdb_url = monetdb_url
            print(f"(Node::__init__) monetdb_url->{monetdb_url}")

            self.__context_id = context_id

            self.task_signatures_str = {
                "get_table": "mipengine.node.tasks.tables.get_tables",
                "get_table_schema": "mipengine.node.tasks.tables.get_table_schema",
                "get_table_data": "mipengine.node.tasks.tables.get_table_data",
                "create_table": "mipengine.node.tasks.tables.create_table",

                "get_views": "mipengine.node.tasks.views.get_views",
                "get_view_schema": "mipengine.node.tasks.views.get_view_schema", #to be removed
                "get_view_data": "mipengine.node.tasks.views.get_view_data", #to be removed
                "create_view": "mipengine.node.tasks.views.create_view",

                "get_remote_tables": "mipengine.node.tasks.remote_tables.get_remote_tables",
                "create_remote_table": "mipengine.node.tasks.remote_tables.create_remote_table",

                "get_merge_tables": "mipengine.node.tasks.merge_tables.get_merge_tables",
                "create_merge_table": "mipengine.node.tasks.merge_tables.create_merge_table",

                "get_udfs": "mipengine.node.tasks.udfs.get_udfs",
                "run_udf": "mipengine.node.tasks.udfs.run_udf",
                "get_udf": "mipengine.node.tasks.udfs.get_udf",
            }

            if base_view_table_params is not None:
                self.__base_view_table_task = self.__create_base_view_table(base_view_table_params)
            self.base_view_table = None

            print("(Node::__init__) finished")

        @property
        def base_view_table(self):
            if self.__base_view_table_task is not None:
                return self.__base_view_table_task.get()  # (timeout=TASK_TIMEOUT)
            else:
                return None

        def __create_base_view_table(self, base_view_table_params):
            print(f"(Node::__create_base_view_table) base_view_table_params->{base_view_table_params}")
            # the base_view_table_name will be the view where the algorithm will do its thing on..
            # table_view_params = TableView(datasets=base_view_table_params["datasets"],
            #                               columns=base_view_table_params["columns"],
            #                               filter="theFuckIsThatSupposedToBe")
            return self.create_view(command_id=base_view_table_params["commandId"],
                                    pathology=base_view_table_params["pathology"],
                                    datasets=base_view_table_params["datasets"],
                                    columns=base_view_table_params["columns"],
                                    filters=base_view_table_params["filters"])

        @base_view_table.setter
        def base_view_table(self, value):
            print(f"(Node::base_view_table) value->{value}")
            self._base_view_name = value

        # TABLES
        def get_tables(self):  # -> List[str]
            task_signature = self.__celery_obj.signature(self.task_signatures_str["get_tables"])
            return task_signature.delay(context_id=self.__context_id)

        def get_table_schema(self, table_name: str):  # -> List["ColumnInfo"]
            task_signature = self.__celery_obj.signature(self.task_signatures_str["get_table_schema"])
            return task_signature.delay(table_name=table_name)

        def get_table_data(self, table_name: str):  # -> "TableData"
            table_name = table_name + "_" + self.node_id  # TODO this was just for testing
            print(f"(Node::get_table_data) table_name->{table_name}")
            task_signature = self.__celery_obj.signature(self.task_signatures_str["get_table_data"])
            return task_signature.delay(table_name=table_name)

        def create_table(self, command_id: str, schema: "TableSchema"):  # ->str
            schema_json = schema.to_json()
            task_signature = self.__celery_obj.signature(self.task_signatures_str["create_table"])
            return task_signature.delay(context_id=self.__context_id, schema_json=schema_json)

        # VIEWS
        def get_views(self):  # -> List[str]:
            task_signature = self.__celery_obj.signature(self.task_signatures_str["get_views"])
            return task_signature.delay(context_id=self.__context_id)

        def create_view(self, command_id: str, pathology: str, datasets: List[str], columns: List[str], filters: List[str]):  # -> "TableInfo":
            print(f"(Node::create_view) columns->{columns} datasets->{datasets} command_id->{command_id}")
            pathology_json = pathology  # json.dumps(pathology)
            columns_json = json.dumps(columns)  # columns.to_json()
            datasets_json = json.dumps(datasets)  # datasets.to_json()
            filters_json = json.dumps(filters)
            task_signature = self.__celery_obj.signature(self.task_signatures_str["create_view"])
            task = task_signature.delay(context_id=self.__context_id,
                                        command_id=command_id,
                                        pathology=pathology_json,
                                        datasets_json=datasets_json,
                                        columns_json=columns_json,
                                        filters_json=filters_json)
            return task

        def get_view_data(self, table_name: str):  # -> "TableData"
            table_name = table_name + "_" + self.node_id  # TODO this was just for testing
            task_signature = self.__celery_obj.signature(self.task_signatures_str["get_view_data"])
            return task_signature.delay(view_name=table_name)

        # MERGE TABLES
        def get_merge_tables(self):  # -> List["TableInfo"]:
            task_signature = self.__celery_obj.signature(self.task_signatures_str["get_merge_tables"])
            return task_signature.delay(context_id=self.__context_id)

        def create_merge_table(self, command_id: str, table_names: List[str]):
            table_names_json = table_names.to_json()
            task_signature = self.__celery_obj.signature(self.task_signatures_str["create_merge_table"])
            return task_signature.delay(context_id=self.__context_id, partition_table_names_json=table_names_json)

        # REMOTE TABLES
        def get_remote_tables(self) -> List["TableInfo"]:
            task_signature = self.__celery_obj.signature(self.task_signatures_str["get_remote_tables"])
            return task_signature.delay(context_id=self.__context_id)

        def create_remote_table(self, command_id: str, table_info: TableInfo, native_node: Node):  # noqa: F821
            table_info_json = table_info.to_json()
            monetdb_url = native_node.monetdb_url
            task_signature = self.__celery_obj.signature(self.task_signatures_str["get_remote_tables"])
            return task_signature.delay(table_info_json=table_info_json, url=monetdb_url)

        # UDFs
        def get_udfs(self):  # -> List["UDFInfo"]:
            task_signature = self.__celery_obj.signature(self.task_signatures_str["get_udfs"])
            return task_signature.delay()

        # return the generated monetdb pythonudf
        def validate_udf(self, udf_input: UDFInput):  # -> str:
            udf_input_json = udf_input.to_json()
            task_signature = self.__celery_obj.signature(self.task_signatures_str["validate_udf"])
            return task_signature.delay(udf_input_json)

        def run_udf(self, command_id: str, udf_input: UDFInput):  # -> "TableInfo":
            udf_input_json = udf_input.to_json()
            task_signature = self.__celery_obj.signature(self.task_signatures_str["run_udf"])
            return task_signature.delay(self.__context_id, udf_input_json)

        def get_udf(self, udf_name: str):  # -> "UDFInfo":
            task_signature = self.__celery_obj.signature(self.task_signatures_str["get_udf"])
            return task_signature.delay(udf_name)

    class AlgorithmExecutionInterface:

        def __init__(self, global_node, local_nodes, algorithm_params):
            self.global_node = global_node
            self.local_nodes = local_nodes
            # TODO: validate all local nodes have created the base_view_table??
            local_views = {}
            for node in self.local_nodes:
                local_views[node] = node.base_view_table
            self._base_view_table = self.LocalNodeTable(nodes_tables=local_views)

        @property
        def base_view_table(self):
            return self._base_view_table

        # UDF calling functionality
        def run_udf_on_local_nodes(self, udf_name: str, udf_input: "UDFInput", share_to_global: bool):  # -> GlobalNodeTable or LocalNodeTable
            # TREAT ALL NODES AS BEEING ONE NODE???
            # <tableType>_<commandId>_<contextIdentifier> to the algorithm developper?

            # check the input tables are LocalNodeTable(s)
            # queue exec_udf task on all local nodes
            # wait for all nodes to complete the tasks execution
            # one new table per local node was generated
            # queue create_remote_table on global for each of the generated tables
            # create merge table on global node to merge the remote tables

            tasks = {}
            for node in self._local_nodes:
                task = node.run_udf(udf_name=udf_name, udf_input=udf_input)
                tasks[node] = task

            # udf_result_tables = []
            udf_result_tables = {}
            for node, task in tasks.items():
                # udf_result_tables[node] = task.get()
                table_info = TableInfo.from_json(task.get())
                udf_result_tables[node] = table_info.name

                # ceate remote table on global node
                if share_to_global:
                    # TODO: try block missing
                    self.global_node.create_remote_table(table_info=table_info, native_node=node).get()

            # create merge table on global
            if share_to_global:
                merge_table_global = self.global_node.create_merge_table(udf_result_tables).get()
                return self.GlobalNodeTable(node=self.global_node, table_name=merge_table_global)

            else:
                return self.LocalNodeTable(nodes_tables=udf_result_tables)

        def run_udf_on_global_node(self, udf_name: str, udf_input: "UDFInput", share_to_locals: bool):  # -> GlobalNodeTable or LocalNodeTable
            # check the input tables are GlobalNodeTable(s)
            # queue exec_udf on the global node
            # wait for it to complete
            # a new table was generated on global node
            # queue create_remote_table on each of the local nodes for the ganerated table

            # TODO: try/catches tasks can throw exceptions
            udf_result_table = self.global_node.run_udf(udf_name=udf_name, udf_input=udf_input).get()
            if share_to_locals:
                local_nodes_tables = {}
                for node in self.local_nodes:
                    local_nodes_tables[node] = node.create_remote_table(table_info=udf_result_table, native_node=self.global_node).get()

                return self.LocalNodeTable(nodes_tables=local_nodes_tables)

            return self.GlobalNodeTable(node=self.global_node, table_name=udf_result_table)

        # TABLES functionality
        def get_table_data_from_global(self, table_name: str) -> "TableData":
            self.global_node.get_table_data(table_name).get()

        class LocalNodeTable():
            def __init__(self, nodes_tables: dict[Node, TableName]):  # noqa: F821
                if self._validate_matching_table_names(list(nodes_tables.values())):
                    self._nodes_tables = nodes_tables
                else:
                    raise self.MismatchingTableNamesException(list(nodes_tables.values()))
                self.__table_name = self._nodes_tables.values()[0].without_node_id()

            @property
            def table_name(self):
                return self.__table_name

            def _validate_matching_table_names(self, table_names: TableName):  # noqa: F821
                table_name_without_node_id = table_names[0].without_node_part()
                for table_name in table_names:
                    if table_name != table_name_without_node_id:
                        return False
                return True

            class MismatchingTableNamesException(Exception):
                def __init__(self, table_names):
                    self.message = f"Mismatched table names ->{table_names}"

        class GlobalNodeTable():
            def __init__(self, node, table_name):  # noqa: F821
                self.__node = node
                self.__table_name = table_name.without_node_id()

            @property
            def table_name(self):
                return self.__table_name

        class TableName:
            def __init__(self, table_name):
                self.__full_name = table_name
                full_name_split = self.__full_name.split("_")
                self.__table_type = full_name_split[0]
                self.__command_id = full_name_split[1]
                self.__context_id = full_name_split[2]
                self.__node_id = full_name_split[3]

            @property
            def full_table_name(self):
                return self.__full_name

            @property
            def table_type(self):
                return self.__table_type

            @property
            def command_id(self):
                return self.__command_id

            @property
            def context_id(self):
                return self.__context_id

            @property
            def node_id(self):
                return self.__node_id

            def without_node_id(self):
                return self.__table_type + "_" + self.__command_id + "_" + self.__context_id


class AlgorithmFlow_abstract(ABC):
    def __init__(self, execution_interface):

        self.runtime_interface = execution_interface


class TableNameNodeInconsistencyException(Exception):
    def __init__(self, table_name, table_type, node_id):
        self.message = f"Table with table_name->{table_name} cannot be instantiated as a {table_type} because its node identifier is {node_id}"


def get_a_uniqueid():
    return "{}".format(datetime.datetime.now().microsecond + (random.randrange(1, 100 + 1) * 100000))


def get_package(algorithm):
    mpackage = "algorithms"
    importlib.import_module(mpackage)
    algo = importlib.import_module("." + algorithm, mpackage)
    return algo











      # class Table:
      #       def __init__(self, table_name):
      #           self.__table_name = table_name
      #           table_name_split = table_name.split("_")
      #           self.__table_type = table_name_split[0]
      #           self.__command_id = table_name_split[1]
      #           self.__context_id = table_name_split[2]
      #           self.__node_id = table_name_split[3]

      #       @property
      #       def table_name(self):
      #           return self.__table_name

      #       @property
      #       def table_type(self):
      #           return self.__table_type

      #       @property
      #       def command_id(self):
      #           return self.__command_id

      #       @property
      #       def context_id(self):
      #           return self.__context_id

      #       @property
      #       def node_id(self):
      #           return self.__node_id

      #       def without_node_Id(self):
      #           return self.__table_type + "_" + self.__command_id + "_" + self.__context_id

      #   class LocalNodeTable(Table):
      #       def __init__(self, table_name):
      #           super().__init__(table_name)
      #           # This would not work, remote tables name keep the nodeId of the native node
      #           # TODO Rethink this convention
      #           # if search("global", self.node_id):
      #           #     exc = TableNameNodeInconsistencyException(self.table_name, self.__class__.__name__, self.node_id)
      #           #     print(exc.message) # TODO remove this, just throw the exception
      #           #     raise exc

      #   class GlobalNodeTable(Table):
      #       def __init__(self, table_name):
      #           super().__init__(table_name)
      #           # This would not work, remote tables name keep the nodeId of the native node
      #           # TODO Rethink this convention
      #           # if not search("global", self.node_id):
      #           #     exc = TableNameNodeInconsistencyException(self.table_name, self.__class__.__name__, self.node_id)
      #           #     print(exc.message) # TODO remove this, just throw the exception
      #           #     raise exc

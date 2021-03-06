from __future__ import annotations
from typing import Dict, List, Any, Optional

import datetime
import random

from abc import ABC

import importlib

from celery import Celery


#  TODO: change those to the actual ones...
from node_catalog import NodeCatalog
from AlgorithmRequestDTO import AlgorithmRequestDTO
from data_classes import ColumnInfo, TableSchema, TableInfo 
from data_classes import TableView, TableData, UDFInfo

import pdb

TASK_TIMEOUT = 2


class AlgorithmExecutor:
    def __init__(self, algorithm_request_dto: AlgorithmRequestDTO):

        print("(AlgorithmExecutor::__init__) just in")

        self.context_id = get_a_uniqueid()  # TODO should this be passed as a param??

        node_catalog = NodeCatalog()
        global_node = node_catalog.get_global_node()
        local_nodes = node_catalog.get_nodes_with_datasets(algorithm_request_dto.datasets)

        # TODO: As is now, if multiple datasets are passed only the nodes of the 1st dataset will be processed
        # TODO: node_catalog returns List[List[Node]] this has to be fixed there
        local_nodes = [node for node in local_nodes[0]]

        # instantiate the GLOBAL Node object
        self.global_node = self.Node(node_id=global_node.nodeId,
                                     rabbitmq_url=global_node.rabbitmqURL,
                                     monetdb_url=f"{global_node.monetdbHostname}:{global_node.monetdbPort}",
                                     context_id=self.context_id)

        initial_view_tables_params = {"commandId": get_next_command_id(),  # get_a_uniqueid(),
                                      "pathology": algorithm_request_dto.pathology,
                                      "datasets": algorithm_request_dto.datasets,
                                      "columns": algorithm_request_dto.columns,
                                      "filters": algorithm_request_dto.filter}

        # instantiate the LOCAL Node objects
        self.local_nodes = []
        for local_node in local_nodes:
            self.local_nodes.append(self.Node(node_id=local_node.nodeId,
                                              rabbitmq_url=local_node.rabbitmqURL,
                                              monetdb_url=f"{local_node.monetdbHostname}:{global_node.monetdbPort}",
                                              initial_view_tables_params=initial_view_tables_params,
                                              context_id=self.context_id))

        execution_interface = self.AlgorithmExecutionInterface(global_node=self.global_node,
                                                               local_nodes=self.local_nodes)
                                                               #algorithm_params=algorithm_request)

        # import the algorithm flow module
        algorithm_folder = algorithm_request_dto.algorithm_name
        algorithm_flow_file = algorithm_request_dto.algorithm_name
        print(f"will import-> algorithm_flows.{algorithm_folder}.{algorithm_flow_file}")
        algorithm_flow_module = importlib.import_module(f"algorithm_flows.{algorithm_folder}.{algorithm_flow_file}_flow")
        print(f"(AlgorithmExecutor::__init__) algorithm_flow_module->{algorithm_flow_module}")

        algorithm_flow_module.AlgorithmFlow(execution_interface).run()
        #algorithm_flow_module.run(execution_interface)
        

        # self.clean_up()

    def clean_up(self):
        self.global_node.clean_up()
        [node.clean_up() for node in self.local_nodes]

    class Node:
        def __init__(self, node_id, rabbitmq_url, monetdb_url, context_id, initial_view_tables_params=None):

            self.node_id = node_id

            # TODO: user, pass, vhost how these should be set??
            user = "user"
            password = "password"
            vhost = "user_vhost"
            broker = f"amqp://{user}:{password}@{rabbitmq_url}/{vhost}"
            self.__celery_obj = Celery(broker=broker, backend='rpc://')

            self.monetdb_url = monetdb_url

            self.__context_id = context_id

            self.task_signatures_str = {
                "get_table": "mipengine.node.tasks.tables.get_tables",
                "get_table_schema": "mipengine.node.tasks.common.get_table_schema",
                "get_table_data": "mipengine.node.tasks.common.get_table_data",
                "create_table": "mipengine.node.tasks.tables.create_table",

                "get_views": "mipengine.node.tasks.views.get_views",
                "create_view": "mipengine.node.tasks.views.create_view",

                "get_remote_tables": "mipengine.node.tasks.remote_tables.get_remote_tables",
                "create_remote_table": "mipengine.node.tasks.remote_tables.create_remote_table",

                "get_merge_tables": "mipengine.node.tasks.merge_tables.get_merge_tables",
                "create_merge_table": "mipengine.node.tasks.merge_tables.create_merge_table",

                "get_udfs": "mipengine.node.tasks.udfs.get_udfs",
                "run_udf": "mipengine.node.tasks.udfs.run_udf",
                "get_udf": "mipengine.node.tasks.udfs.get_udf",

                "clean_up": "mipengine.node.tasks.common.clean_up"
            }

            self.__initial_view_tables = None
            if initial_view_tables_params is not None:
                self.__initial_view_tables = self.__create_initial_view_tables(initial_view_tables_params)

        @property
        def initial_view_tables(self):
            return self.__initial_view_tables

        def __create_initial_view_tables(self, initial_view_tables_params):
            print(f"initial_view_tables_params->{initial_view_tables_params}")
            # the initial_view_table_name will be the view created from the pathology, datasets, x, y etc passed to the algorithm
            initial_view_tables = {}
            for variables_set in initial_view_tables_params["columns"]:
                command_id = str(initial_view_tables_params["commandId"]) + variables_set.name
                view_name = self.create_view(command_id=command_id,
                                             pathology=initial_view_tables_params["pathology"],
                                             datasets=initial_view_tables_params["datasets"],
                                             columns=variables_set.column_names,
                                             filters=initial_view_tables_params["filters"])

                initial_view_tables[variables_set.name] = view_name

            return initial_view_tables

        # TABLES functionality
        def get_tables(self) -> List[TableName]:
            task_signature = self.__celery_obj.signature(self.task_signatures_str["get_tables"])
            result = task_signature.delay(context_id=self.__context_id).get()
            return [self.TableName(table_name) for table_name in result]

        def get_table_schema(self, table_name: TableName):
            task_signature = self.__celery_obj.signature(self.task_signatures_str["get_table_schema"])
            result = task_signature.delay(table_name=table_name.full_table_name).get()
            return TableSchema.from_json(result)

        def get_table_data(self, table_name: TableName) -> TableData:
            task_signature = self.__celery_obj.signature(self.task_signatures_str["get_table_data"])
            result = task_signature.delay(table_name=table_name.full_table_name).get()
            return TableData.from_json(result)

        def create_table(self, command_id: str, schema: TableSchema) -> TableName:
            schema_json = schema.to_json()
            task_signature = self.__celery_obj.signature(self.task_signatures_str["create_table"])
            result = task_signature.delay(context_id=self.__context_id, schema_json=schema_json).get()
            return self.TableName(result)

        # VIEWS functionality
        def get_views(self) -> List[TableName]:
            task_signature = self.__celery_obj.signature(self.task_signatures_str["get_views"])
            result = task_signature.delay(context_id=self.__context_id).get()
            return [self.TableName(table_name) for table_name in result]

        # TODO: this is very specific to mip, very inconsistent with the rest, has to be abstracted somehow
        def create_view(self, command_id: str, pathology: str, datasets: List[str], columns: List[str], filters: List[str]) -> TableName:
            task_signature = self.__celery_obj.signature(self.task_signatures_str["create_view"])
            # -----------DEBUG
            #command_id = get_next_command_id()
            print(f"(Node::create_view) context_id->{self.__context_id} type->{type(self.__context_id)}")
            print(f"(Node::create_view) command_id->{command_id} type->{type(command_id)}")
            print(f"(Node::create_view) pathology->{pathology} type->{type(pathology)}")
            print(f"(Node::create_view) datasets->{datasets} type->{type(datasets)}")
            print(f"(Node::create_view) columns->{columns} type->{type(columns)}")
            print(f"(Node::create_view) filters->{filters} type->{type(filters)}")

            # --------------
            result = task_signature.delay(context_id=self.__context_id,
                                          command_id=command_id,
                                          pathology=pathology,
                                          datasets=datasets,
                                          columns=columns,
                                          filters_json=filters)
            result = result.get()
            return self.TableName(result)

        # MERGE TABLES functionality
        def get_merge_tables(self) -> List[TableName]:
            task_signature = self.__celery_obj.signature(self.task_signatures_str["get_merge_tables"])
            result = task_signature.delay(context_id=self.__context_id).get()
            return [self.TableName(table_name) for table_name in result]

        def create_merge_table(self, command_id: str, table_names: List[TableName]):  # noqa: F821
            table_names = [table_name.full_table_name for table_name in table_names]
            task_signature = self.__celery_obj.signature(self.task_signatures_str["create_merge_table"])
            result = task_signature.delay(command_id=command_id, context_id=self.__context_id, partition_table_names_json=table_names).get()
            return self.TableName(result)

        # REMOTE TABLES functionality
        def get_remote_tables(self) -> List["TableInfo"]:
            task_signature = self.__celery_obj.signature(self.task_signatures_str["get_remote_tables"])
            return task_signature.delay(context_id=self.__context_id)

        def create_remote_table(self, command_id: str, table_info: TableInfo, native_node: Node) -> TableName: # noqa: F821
            table_info_json = table_info.to_json()
            monetdb_url = native_node.monetdb_url
            task_signature = self.__celery_obj.signature(self.task_signatures_str["get_remote_tables"])
            result = task_signature.delay(table_info_json=table_info_json, url=monetdb_url).get()
            return self.TableName(TableInfo(result).name)

        # UDFs functionality
        def queue_run_udf(self, command_id: str, udf_input: UDFInput) -> AsyncResult:
            task_signature = self.__celery_obj.signature(self.task_signatures_str["run_udf"])
            return task_signature.delay(command_id=command_id, context_id=self.__context_id, udf_input=udf_input)

        def get_result_run_udf(self, async_result) -> str:
            result = async_result.get()
            return self.TableName(result)

        def get_udfs(self, algorithm_name) -> List["UDFInfo"]:
            task_signature = self.__celery_obj.signature(self.task_signatures_str["get_udfs"])
            result = task_signature.delay(algorithm_name).get()
            return result

        # return the generated monetdb pythonudf
        def validate_udf(self, command_id: str, udf_input: UDFInput) -> str:
            task_signature = self.__celery_obj.signature(self.task_signatures_str["run_udf"])
            result = task_signature.delay(command_id=command_id, context_id=self.__context_id, udf_input=udf_input).get()
            return result

        #CLEANUP functionality
        def clean_up(self):
            task_signature = self.__celery_obj.signature(self.task_signatures_str["clean_up"])
            task_signature.delay(self.__context_id)

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

    class AlgorithmExecutionInterface:

        def __init__(self, global_node, local_nodes):
            self.global_node = global_node
            self.local_nodes = local_nodes
            # TODO: validate all local nodes have created the base_view_table??

            self._initial_view_tables = {}  # variable:LocalTable
            tmp_node_table_names = {}
            tmp_variable_node_table = {}
            print()
            for node in self.local_nodes:
                print(f"node -> {node.node_id}")
                for (variable_name, table_name) in node.initial_view_tables.items():
                    print(f"\t\tvariable_name -> {variable_name}    table_name->{table_name.full_table_name}")
                    if variable_name in tmp_variable_node_table:
                        tmp_variable_node_table[variable_name].update({node: table_name})
                    else:
                        tmp_variable_node_table[variable_name] = {node: table_name}
                    print(f"tmp_variable_node_table-> {tmp_variable_node_table}\n\n")

            print(f"\n\ntmp_variable_node_table->{tmp_variable_node_table}\n\n")

            self._initial_view_tables = {variable_name: self.LocalNodeTable(node_table) for (variable_name, node_table) in tmp_variable_node_table.items()} 

        @property
        def initial_view_tables(self):
            return self._initial_view_tables

        # UDFs functionality
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
                task = node.run_udf(command_id=get_a_uniqueid(), udf_name=udf_name, udf_input=udf_input)
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
                remote_tables_info = List(udf_result_tables.values())
                remote_table_names = [remote_table_info.name for remote_table_info in remote_tables_info]
                merge_table_global = self.global_node.create_merge_table(command_id=get_a_uniqueid(), table_names=remote_table_names).get()
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
            udf_result_table = self.global_node.run_udf(command_id=get_a_uniqueid(), udf_name=udf_name, udf_input=udf_input).get()

            if share_to_locals:
                local_nodes_tables = {}
                for node in self.local_nodes:
                    # TODO do not block here, first send the request to all local nodes and then block for the result
                    local_nodes_tables[node] = node.create_remote_table(table_info=udf_result_table, native_node=self.global_node).get()

                return self.LocalNodeTable(nodes_tables=local_nodes_tables)

            return self.GlobalNodeTable(node=self.global_node, table_name=udf_result_table)

        # TABLES functionality
        def get_table_data_from_global(self, table_name: TableName) -> "TableData":
            return self.global_node.get_table_data(table_name)

        # DEBUG 
        def get_table_data_from_local(self, node: Node,table_name: TableName) -> "TableData":
            return node.get_table_data(table_name)

        class LocalNodeTable():
            def __init__(self, nodes_tables: dict[Node, TableName]):  # noqa: F821
                self.__nodes_tables = nodes_tables  # {node: self.TableName(table_name) for (node, table_name) in nodes_tables.items()}

                if not self._validate_matching_table_names(list(self.__nodes_tables.values())):
                    raise self.MismatchingTableNamesException(list(nodes_tables.values()))

            @property
            def nodes_tables(self):
                return self.__nodes_tables

            def _validate_matching_table_names(self, table_names: List[TableName]):  # noqa: F821
                table_name_without_node_id = table_names[0].without_node_id()
                for table_name in table_names:
                    if table_name.without_node_id() != table_name_without_node_id:
                        return False
                return True

            class MismatchingTableNamesException(Exception):
                def __init__(self, table_names):
                    self.message = f"Mismatched table names ->{table_names}"

        class GlobalNodeTable():
            def __init__(self, node, table_name: TableName):  # noqa: F821
                self.__node = node
                self.__table_name = TableName(table_name).without_node_id()  # noqa: F821

            @property
            def table_name(self):
                return self.__table_name

            @property
            def node(self):
                return self.__node


class AlgorithmFlow_abstract(ABC):
    def __init__(self, execution_interface):
        self.runtime_interface = execution_interface


def get_a_uniqueid():
    return "{}".format(datetime.datetime.now().microsecond + (random.randrange(1, 100 + 1) * 100000))


def get_next_command_id():
    if hasattr(get_next_command_id, "index"):
        get_next_command_id.index += 1
    else:
        get_next_command_id.index = 0
    return get_next_command_id.index

def get_package(algorithm):
    mpackage = "algorithms"
    importlib.import_module(mpackage)
    algo = importlib.import_module("." + algorithm, mpackage)
    return algo
import warnings
from abc import ABC
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import pandas as pd

from exareme2 import DATA_TABLE_PRIMARY_KEY
from exareme2.controller.services.exareme2.workers import GlobalWorker
from exareme2.controller.services.exareme2.workers import LocalWorker
from exareme2.worker_communication import SMPCTablesInfo
from exareme2.worker_communication import TableInfo
from exareme2.worker_communication import TableSchema
from exareme2.worker_communication import WorkerLiteralDTO
from exareme2.worker_communication import WorkerSMPCDTO
from exareme2.worker_communication import WorkerTableDTO
from exareme2.worker_communication import WorkerUDFDTO
from exareme2.worker_communication import WorkerUDFKeyArguments
from exareme2.worker_communication import WorkerUDFPosArguments


class AlgoFlowData(ABC):
    """
    AlgoFlowData are representing data objects in the algorithm flow.
    These objects are the result of running udfs and are used as input
    as well in the udfs.
    """

    _schema: TableSchema

    @property
    def full_schema(self):
        """
        Returns the full schema of the table, index + column names in a `TableSchema` format.
        """
        return self._schema

    @property
    def index(self):
        """
        Returns the index of the table schema if one exists.
        """
        if DATA_TABLE_PRIMARY_KEY in self._schema.column_names:
            return DATA_TABLE_PRIMARY_KEY
        else:
            return None

    @property
    def columns(self):
        """
        Returns the columns of the table without the index.
        """
        return [
            column.name
            for column in self._schema.columns
            if column.name != DATA_TABLE_PRIMARY_KEY
        ]


class LocalWorkersData(AlgoFlowData, ABC):
    """
    LocalWorkersData are representing data objects in the algorithm flow
    that are located in many (or one) local workers.
    """

    pass


class GlobalWorkerData(AlgoFlowData, ABC):
    """
    GlobalWorkerData are representing data objects in the algorithm flow
    that are located in the global worker.
    """

    pass


class LocalWorkersTable(LocalWorkersData):
    """
    A LocalWorkersTable is a representation of a table across multiple workers. To this end,
    it holds references to the actual workers and tables through a dictionary with its keys
    being workers and values being a table on that worker.

    example:
      When Exareme2AlgorithmFlowEngineInterface::run_udf_on_local_workers(..) is called, depending on
    how many local workers are participating in the current algorithm execution, several
    database tables are created on all participating local workers. Irrespective of the
    number of local workers participating, the number of tables created on each of these local
    workers will be the same.
    Class LocalWorkTable is the structure that represents the concept of these database
    tables, created during the execution of a udf, in the algorithm execution layer. A key
    concept is that a LocalWorkTable stores 'pointers' to 'relevant' tables existing in
    different local workers across the federation. What 'relevant' means is that the tables
    are generated when triggering a udf execution across several local workers. What
    'pointers' means is that there is a mapping between local workers and table names and
    the aim is to hide the underline complexity from the algorithm flow and exposing a
    single 'local worker table' object that stores in the background pointers to several
    tables in several local workers.
    """

    _workers_tables_info: Dict[LocalWorker, TableInfo]

    def __init__(self, workers_tables_info: Dict[LocalWorker, TableInfo]):
        self._workers_tables_info = workers_tables_info
        self._validate_matching_table_names()
        self._schema = next(iter(workers_tables_info.values())).schema_

    @property
    def workers_tables_info(self) -> Dict[LocalWorker, TableInfo]:
        return self._workers_tables_info

    def get_table_data(self) -> List[Union[List[int], List[float], List[str]]]:
        """Gets merged data from corresponding table in all workers

        A LocalWorkersTable represents a collection of tables spread across
        multiple workers. This method gets data from all tables and merges it in
        a common tables.

        The `row_id` column is treated differently. In each worker `row_id` is an
        incrementing integer which is unique within each worker, but not across
        workers. Therefor, concatenating `row_id` columns would result in non
        unique values, hence the column would not be a primary key anymore. To
        remedy this, the current method prepends the worker_id to each row_id
        using the format "worker_id:row_id". This guarantees uniqueness of
        values, making `row_id` a primary key of the merged table.
        """

        msg = "LocalWorkersTable.get_table_data should not be used in production."
        warnings.warn(msg)

        dataframe_per_worker = {
            worker.worker_id: worker.get_table_data(table_info.name).to_pandas()
            for worker, table_info in self.workers_tables_info.items()
        }

        # Sort according to worker_id to have a common order for all tables
        sorted_worker_ids = sorted(dataframe_per_worker.keys())
        dataframe_per_worker = {
            worker_id: dataframe_per_worker[worker_id]
            for worker_id in sorted_worker_ids
        }

        # Make multi-index dataframe using both worker_id and row_id
        for worker_id, df in dataframe_per_worker.items():
            df["worker_id"] = worker_id
            df.set_index(["worker_id", "row_id"], inplace=True)

        # Merge dataframes from all workers
        dataframes = list(dataframe_per_worker.values())
        merged_df = pd.concat(dataframes)

        # Convert to List[Union[List[int], List[float], List[str]]] where the
        # row_id column is now composed of strings of the form worker_id:row_id
        index = merged_df.index.tolist()
        index = [f"{worker_id}:{row_id}" for worker_id, row_id in index]
        data = merged_df.T.values.tolist()
        return [index] + data

    def __repr__(self):
        r = "LocalWorkTable:\n"
        for worker, table_info in self.workers_tables_info.items():
            r += f"\t{worker=} {table_info=}\n"
        return r

    def _validate_matching_table_names(self):
        table_infos = list(self._workers_tables_info.values())
        table_name_without_worker_id = table_infos[0].name_without_worker_id
        for table_name in table_infos:
            if table_name.name_without_worker_id != table_name_without_worker_id:
                raise MismatchingTableNamesException(
                    [table_info.name for table_info in table_infos]
                )


class GlobalWorkerTable(GlobalWorkerData):
    _worker: GlobalWorker
    _table_info: TableInfo

    def __init__(self, worker: GlobalWorker, table_info: TableInfo):
        self._worker = worker
        self._table_info = table_info
        self._schema = table_info.schema_

    @property
    def worker(self) -> GlobalWorker:
        return self._worker

    @property
    def table_info(self) -> TableInfo:
        return self._table_info

    def get_table_data(self) -> List[List[Any]]:
        table_data = [
            column.data
            for column in self.worker.get_table_data(self.table_info.name).columns
        ]
        return table_data

    def __repr__(self):
        r = f"\n\tGlobalWorkerTable: \n\t{self._schema=}\n \t{self.table_info=}\n"
        return r


class LocalWorkersSMPCTables(LocalWorkersData):
    _smpc_tables_info_per_worker: Dict[LocalWorker, SMPCTablesInfo]

    def __init__(self, smpc_tables_info_per_worker: Dict[LocalWorker, SMPCTablesInfo]):
        self._smpc_tables_info_per_worker = smpc_tables_info_per_worker

    @property
    def workers_smpc_tables(self) -> Dict[LocalWorker, SMPCTablesInfo]:
        return self._smpc_tables_info_per_worker

    @property
    def template_local_workers_table(self) -> LocalWorkersTable:
        return LocalWorkersTable(
            {
                worker: tables.template
                for worker, tables in self.workers_smpc_tables.items()
            }
        )

    @property
    def sum_op_local_workers_table(self) -> Optional[LocalWorkersTable]:
        workers_tables = {}
        for worker, tables in self.workers_smpc_tables.items():
            if not tables.sum_op:
                return None
            workers_tables[worker] = tables.sum_op
        return LocalWorkersTable(workers_tables)

    @property
    def min_op_local_workers_table(self) -> Optional[LocalWorkersTable]:
        workers_tables = {}
        for worker, tables in self.workers_smpc_tables.items():
            if not tables.min_op:
                return None
            workers_tables[worker] = tables.min_op
        return LocalWorkersTable(workers_tables)

    @property
    def max_op_local_workers_table(self) -> Optional[LocalWorkersTable]:
        workers_tables = {}
        for worker, tables in self.workers_smpc_tables.items():
            if not tables.max_op:
                return None
            workers_tables[worker] = tables.max_op
        return LocalWorkersTable(workers_tables)


class GlobalWorkerSMPCTables(GlobalWorkerData):
    _worker: GlobalWorker
    _smpc_tables_info: SMPCTablesInfo

    def __init__(self, worker: GlobalWorker, smpc_tables_info: SMPCTablesInfo):
        self._worker = worker
        self._smpc_tables_info = smpc_tables_info

    @property
    def worker(self) -> GlobalWorker:
        return self._worker

    @property
    def smpc_tables_info(self) -> SMPCTablesInfo:
        return self._smpc_tables_info


def algoexec_udf_kwargs_to_worker_udf_kwargs(
    algoexec_kwargs: Dict[str, Any],
    local_worker: LocalWorker = None,
) -> WorkerUDFKeyArguments:
    if not algoexec_kwargs:
        return WorkerUDFKeyArguments(args={})

    args = {}
    for key, arg in algoexec_kwargs.items():
        udf_argument = _algoexec_udf_arg_to_worker_udf_arg(arg, local_worker)
        args[key] = udf_argument
    return WorkerUDFKeyArguments(args=args)


def algoexec_udf_posargs_to_worker_udf_posargs(
    algoexec_posargs: List[Any],
    local_worker: LocalWorker = None,
) -> WorkerUDFPosArguments:
    if not algoexec_posargs:
        return WorkerUDFPosArguments(args=[])

    args = []
    for arg in algoexec_posargs:
        args.append(_algoexec_udf_arg_to_worker_udf_arg(arg, local_worker))
    return WorkerUDFPosArguments(args=args)


def _algoexec_udf_arg_to_worker_udf_arg(
    algoexec_arg: AlgoFlowData, local_worker: LocalWorker = None
) -> WorkerUDFDTO:
    """
    Converts the algorithm executor run_udf input arguments, coming from the algorithm flow
    to worker udf pos/key arguments to be sent to the WORKER.

    Parameters
    ----------
    algoexec_arg is the argument to be converted.
    local_worker is need only when the algoexec_arg is of LocalWorkersTable, to know
                which local table should be selected.

    Returns
    -------
    a WorkerUDFDTO
    """
    if isinstance(algoexec_arg, LocalWorkersTable):
        if not local_worker:
            raise ValueError(
                "local_worker parameter is required on LocalWorkersTable conversion."
            )
        return WorkerTableDTO(value=algoexec_arg.workers_tables_info[local_worker])
    elif isinstance(algoexec_arg, GlobalWorkerTable):
        return WorkerTableDTO(value=algoexec_arg.table_info)
    elif isinstance(algoexec_arg, LocalWorkersSMPCTables):
        raise ValueError(
            "'LocalWorkersSMPCTables' cannot be used as argument. It must be shared."
        )
    elif isinstance(algoexec_arg, GlobalWorkerSMPCTables):
        return WorkerSMPCDTO(value=algoexec_arg.smpc_tables_info)
    else:
        return WorkerLiteralDTO(value=algoexec_arg)


def create_worker_table_dto_from_global_worker_table(table_info: TableInfo):
    if not table_info:
        return None

    return WorkerTableDTO(value=table_info)


class MismatchingTableNamesException(Exception):
    def __init__(self, table_names: List[str]):
        message = f"Mismatched table names ->{table_names}"
        super().__init__(message)
        self.message = message

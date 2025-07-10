import itertools
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional

from exareme2.controller.services.exareme2.algorithm_flow_data_objects import (
    LocalWorkersTable,
)
from exareme2.controller.services.exareme2.workers import LocalWorker
from exareme2.worker_communication import InsufficientDataError
from exareme2.worker_communication import TableInfo


class DataModelViews:
    def __init__(self, local_worker_tables: Iterable[LocalWorkersTable]):
        self._views = local_worker_tables

    @classmethod
    def from_views_per_localworker(
        cls, views_per_localworker: Dict[LocalWorker, List[TableInfo]]
    ):
        return cls(
            cls._views_per_localworker_to_localworkerstables(views_per_localworker)
        )

    def to_list(self):
        return self._views

    def get_list_of_workers(self) -> List[LocalWorker]:
        """
        LocalWorkersTable is representation of a table across multiple workers. A DataModelView
        consists a collection of LocalWorkersTables that might exist on different set
        of workers or even on overlapping set of workers. This method returns a list of all
        the workers in the collection of LocalWorkersTables without duplicates
        """
        workers = set()
        for local_workers_table in self._views:
            workers.update(local_workers_table.workers_tables_info.keys())
        if not workers:
            raise InsufficientDataError(
                "None of the workers has enough data to execute the algorithm."
            )
        return list(workers)

    @classmethod
    def _views_per_localworker_to_localworkerstables(
        cls, views_per_localworker: Dict[LocalWorker, List[TableInfo]]
    ) -> List[LocalWorkersTable]:
        """
        Combines the tables of different workers into LocalWorkersTables
        """
        number_of_tables = cls._validate_number_of_views(views_per_localworker)

        local_workers_tables = [
            LocalWorkersTable(
                {worker: tables[i] for worker, tables in views_per_localworker.items()}
            )
            for i in range(number_of_tables)
        ]

        return local_workers_tables

    @classmethod
    def _validate_number_of_views(
        cls, views_per_localworker: Dict[LocalWorker, List[TableInfo]]
    ):
        """
        Checks that the number of views is the same for all workers
        """
        number_of_tables = [len(tables) for tables in views_per_localworker.values()]
        number_of_tables_equal = len(set(number_of_tables)) == 1

        if not number_of_tables_equal:
            raise ValueError(
                "The number of views is not the same for all workers"
                f" {views_per_localworker=}"
            )
        return number_of_tables[0]


class DataModelViewsCreator:
    """
    Choosing which subset of the connected to the system workers will participate in an
    execution takes place in 2 steps. The first step is choosing the workers containing
    the "data model" and datasets in the request. The second is when "data model views"
    are created. During this second step, depending on the "minimum_row_count" threshold,
    a worker that was chosen in the previous step might be left out because of
    "insufficient data". So the procedure of creating the, so called, "data model views"
    plays a key role in defining the subset of the workers that will participate in an
    execution. The DataModelCreator class implements the aforementioned functionality,
    of generating a DataModelView object as well as identifying the subset of workers
    eligible for a specific execution request.
    """

    def __init__(
        self,
        local_workers: List[LocalWorker],
        variable_groups: List[List[str]],
        var_filters: list,
        dropna: bool,
        check_min_rows: bool,
        command_id: int,
    ):
        """
        Parameters
        ----------
        local_workers: List[LocalWorker]
            The list of LocalWorkers on which the "data model views" will be created(if
            there is "sufficient data")
        variable_groups: List[List[str]]
            The variable groups
        var_filters: list
            The filtering parameters
        dropna: bool
            A boolean flag denoting if the 'Not Available' values will be kept in the
            "data model views" or not
        check_min_rows: bool
            A boolean flag denoting if a "minimum row count threshold" will be in place
            or not
        command_id: int
            A unique id
        """
        self._local_workers = local_workers
        self._variable_groups = variable_groups
        self._var_filters = var_filters
        self._dropna = dropna
        self._check_min_rows = check_min_rows
        self._command_id = command_id

        self._data_model_views = None

    @property
    def data_model_views(self):
        return self._data_model_views

    def create_data_model_views(
        self,
    ) -> Optional[DataModelViews]:
        """
        Creates the "data model views", for each variable group provided,
        using also the algorithm request arguments (data_model, datasets, filters).

        Returns
        ------
        DataModelViews
            A DataModelViews containing a view for each variable_group provided.
        """

        if self._data_model_views:
            return

        if not list(itertools.chain(*self._variable_groups)):
            raise ValueError(
                "There are not variables in the 'variable_groups' of the algorithm. "
                "Please check that the 'variable_groups' in the data loader are pointing to the proper variables."
            )

        views_per_localworker = {}
        for worker in self._local_workers:
            try:
                data_model_views = worker.create_data_model_views(
                    command_id=self._command_id,
                    columns_per_view=self._variable_groups,
                    filters=self._var_filters,
                    dropna=self._dropna,
                    check_min_rows=self._check_min_rows,
                )
            except InsufficientDataError:
                continue
            views_per_localworker[worker] = data_model_views

        if not views_per_localworker:
            raise InsufficientDataError(
                "None of the workers has enough data to execute request: {LocalWorkers:"
                "Datasets}-> "
                f"{ {worker.worker_id:worker.datasets for worker in self._local_workers} } "
                f"{self._variable_groups=} {self._var_filters=} {self._dropna=} "
                f"{self._check_min_rows=}"
            )

        self._data_model_views = DataModelViews.from_views_per_localworker(
            views_per_localworker
        )

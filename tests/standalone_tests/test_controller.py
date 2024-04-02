from dataclasses import dataclass
from typing import List
from typing import Tuple
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from exareme2.controller.celery.tasks_handlers import Exareme2TasksHandler
from exareme2.controller.services.exareme2.algorithm_flow_data_objects import (
    LocalWorkersTable,
)
from exareme2.controller.services.exareme2.controller import DataModelViews
from exareme2.controller.services.exareme2.controller import DataModelViewsCreator
from exareme2.controller.services.exareme2.controller import WorkersFederation
from exareme2.controller.services.exareme2.execution_engine import Workers
from exareme2.controller.services.exareme2.workers import LocalWorker
from exareme2.worker_communication import InsufficientDataError
from exareme2.worker_communication import TableInfo
from exareme2.worker_communication import TableSchema
from exareme2.worker_communication import TableType


def create_dummy_worker(worker_id: str, context_id: str, request_id: str):
    return LocalWorker(
        request_id=request_id,
        context_id=context_id,
        exareme2_tasks_handler=Exareme2TasksHandler("0", worker_id, "0", "0", 10, 10),
        data_model="",
        datasets=[],
    )


@pytest.fixture
def worker_mocks():
    # context_id = "0"
    workers_ids = ["worker" + str(i) for i in range(1, 11)]
    workers = [
        LocalWorker(
            request_id="0",
            context_id="0",
            exareme2_tasks_handler=Exareme2TasksHandler(
                "0", worker_id, "0", "0", 10, 10
            ),
            data_model="",
            datasets=[],
        )
        for worker_id in workers_ids
    ]
    return workers


class TestWorkersFederation:
    class WorkerInfoMock:
        def __init__(self, worker_id: str):
            self.id = worker_id
            self.ip = ""
            self.port = ""
            self.db_ip = ""
            self.db_port = ""

    class WorkerLandscapeAggregatorMock:
        @property
        def workerids_datasets_mock(self):
            return {
                "worker1": ["dataset1", "dataset3", "dataset4"],
                "worker2": ["dataset2", "dataset8", "dataset5"],
                "worker3": ["dataset9", "dataset6", "dataset7"],
            }

        @property
        def globalworkerid(self):
            return "globalworker"

        def get_worker_ids_with_any_of_datasets(self, *args, **kwargs):
            return list(self.workerids_datasets_mock.keys())

        def get_worker_info(self, worker_id: str):
            return TestWorkersFederation.WorkerInfoMock(worker_id)

        def get_worker_specific_datasets(
            self, worker_id: str, data_model: str, wanted_datasets: List[str]
        ):
            return self.workerids_datasets_mock[worker_id]

        def get_global_worker(self):
            return TestWorkersFederation.WorkerInfoMock(self.globalworkerid)

    class CommandIdGeneratorMock:
        pass

    class LoggerMock:
        def debug(self, *args, **kwargs):
            pass

        def debug(self, *args, **kwargs):
            pass

    @pytest.fixture
    def workers_federation_mock(self):
        return WorkersFederation(
            request_id="0",
            context_id="0",
            data_model="",
            datasets=[""],
            var_filters={},
            worker_landscape_aggregator=self.WorkerLandscapeAggregatorMock(),
            celery_tasks_timeout=0,
            celery_run_udf_task_timeout=0,
            command_id_generator=self.CommandIdGeneratorMock(),
            logger=self.LoggerMock(),
        )

    # def test_get_workerinfo_for_requested_datasets(self):
    #     pass

    # def test_worker_ids(self):
    #     pass

    # def test_create_data_model_views(self):
    #     pass

    def test_create_workers(self, workers_federation_mock):
        workers_federation = workers_federation_mock
        created_workers = workers_federation._create_workers()

        assert isinstance(created_workers, Workers)

        created_localworkerids = [
            localworker.worker_id for localworker in created_workers.local_workers
        ]
        created_globalworkerid = created_workers.global_worker.worker_id

        expected_localworkerids = (
            workers_federation._worker_landscape_aggregator.workerids_datasets_mock.keys()
        )
        assert all(
            workerid in created_localworkerids for workerid in expected_localworkerids
        )

        expected_globalworkerid = (
            workers_federation._worker_landscape_aggregator.globalworkerid
        )
        assert expected_globalworkerid == created_globalworkerid

    # def test_get_datasets_of_workerids(self):
    #     pass

    # def test_create_workers_tasks_handlers(self):
    #     pass

    # def test_get_workers_info(self):
    #     pass


class TestDataModelViews:
    @pytest.fixture
    def views_mocks(self, worker_mocks):
        # table naming convention <table_type>_<worker_id>_<context_id>_<command_id>_<result_id>
        table_info = TableInfo(
            name=str(TableType.NORMAL).lower()
            + "_"
            + local_worker.worker_id
            + "_0_"
            + context_id
            + "_0",
            schema_=schema,
            type_=TableType.NORMAL,
        )
        views = [LocalWorkersTable(workers_tables_info={worker_mocks[0], table_info})]
        return views

    def test_get_workers(self):
        class LocalWorkersTable:
            def __init__(self, worker_ids: List[str]):
                self.workers_tables_info = {worker_id: None for worker_id in worker_ids}

        # create a DataModelViews object that contains some LocalWorkersTables
        local_worker_tables_mock = [
            LocalWorkersTable(["worker1", "worker2", "worker3"]),
            LocalWorkersTable(["worker1", "worker8", "worker3"]),
            LocalWorkersTable(["worker1", "worker3", "worker2"]),
            LocalWorkersTable(["worker3", "worker1", "worker8"]),
            LocalWorkersTable(["worker2", "worker1", "worker3"]),
        ]
        data_model_views = DataModelViews(local_worker_tables_mock)
        result = data_model_views.get_list_of_workers()

        # get all worker_ids from local_worker_tables_mock
        tmp = [
            local_worker_table.workers_tables_info.keys()
            for local_worker_table in local_worker_tables_mock
        ]
        tmp_lists = [list(t) for t in tmp]
        expected_worker_ids = set(
            [worker_id for sublist in tmp_lists for worker_id in sublist]
        )

        assert all(worker_id in result for worker_id in expected_worker_ids)

        @pytest.fixture
        def views_per_local_workers():
            tables_views = {
                "worker1": ["view1_1", "view1_2"],
                "worker2": ["view2_1", "view2_2"],
                "worker3": ["view3_1", "view3_2"],
            }
            return tables_views

        @pytest.fixture
        def workers_tables_expected():
            expected = [
                {"worker1": "view1_1", "worker2": "view2_1", "worker3": "view3_1"},
                {"worker1": "view1_2", "worker2": "view2_2", "worker3": "view3_2"},
            ]
            return expected

        @pytest.fixture
        def views_per_local_workers_invalid():
            tables_views = {
                "worker1": ["view1_1", "view1_2"],
                "worker2": ["view2_1"],
                "worker3": ["view3_1", "view3_2"],
            }
            return tables_views

        def test_validate_number_of_views(
            views_per_local_workers, views_per_local_workers_invalid
        ):
            tableinfo_list = list(views_per_local_workers.values())
            assert DataModelViews._validate_number_of_views(
                views_per_local_workers
            ) == len(tableinfo_list[0])

            with pytest.raises(ValueError):
                DataModelViews._validate_number_of_views(
                    views_per_local_workers_invalid
                )

        def test_views_per_localworker_to_localworkerstables(
            views_per_local_workers, workers_tables_expected
        ):
            class MockLocalWorkersTable:
                def __init__(self, workers_tables_info: dict):
                    self._workers_tables_info = workers_tables_info

            with patch(
                "exareme2.controller.controller.LocalWorkersTable",
                MockLocalWorkersTable,
            ):
                local_workers_tables = (
                    DataModelViews._views_per_localworker_to_localworkerstables(
                        views_per_local_workers
                    )
                )

            workers_tables_info = [t._workers_tables_info for t in local_workers_tables]
            for expected in workers_tables_expected:
                assert expected in workers_tables_info

                assert len(workers_tables_expected) == len(workers_tables_info)


class TestDataModelViewsCreator:
    class TableInfoMock:
        name: str
        schema_: TableSchema
        type_: TableType

        @property
        def column_names(self):
            pass

        @property
        def _tablename_parts(self) -> Tuple[str, str, str, str]:
            pass

        @property
        def worker_id(self) -> str:
            pass

        @property
        def context_id(self) -> str:
            pass

        @property
        def command_id(self) -> str:
            pass

        @property
        def result_id(self) -> str:
            pass

        @property
        def name_without_worker_id(self) -> str:
            return "workername"

    @pytest.fixture
    def local_worker_mocks(self):
        return [MagicMock(LocalWorker) for number_of_workers in range(10)]

    @pytest.fixture
    def data_model_views_creator_init_params(self, local_worker_mocks):
        @dataclass
        class DataModelViewsCreatorInitParams:
            local_workers: List[LocalWorker]
            variable_groups: List[List[str]]
            var_filters: list
            dropna: bool
            check_min_rows: bool
            command_id: int

        return DataModelViewsCreatorInitParams(
            local_workers=local_worker_mocks,
            variable_groups=[["v1," "v2"], ["v3", "v4"]],
            var_filters=[],
            dropna=False,
            check_min_rows=True,
            command_id=123,
        )

    def test_create_data_model_views_called_on_all_workers(
        self, local_worker_mocks, data_model_views_creator_init_params
    ):
        data_model_views_creator = DataModelViewsCreator(
            local_workers=data_model_views_creator_init_params.local_workers,
            variable_groups=data_model_views_creator_init_params.variable_groups,
            var_filters=data_model_views_creator_init_params.var_filters,
            dropna=data_model_views_creator_init_params.dropna,
            check_min_rows=data_model_views_creator_init_params.check_min_rows,
            command_id=data_model_views_creator_init_params.command_id,
        )

        # assert that create_data_model_views was called for all local workers with the
        # expected args
        data_model_views_creator.create_data_model_views()
        for worker in local_worker_mocks:
            worker.create_data_model_views.assert_called_once_with(
                columns_per_view=data_model_views_creator_init_params.variable_groups,
                filters=data_model_views_creator_init_params.var_filters,
                dropna=data_model_views_creator_init_params.dropna,
                check_min_rows=data_model_views_creator_init_params.check_min_rows,
                command_id=data_model_views_creator_init_params.command_id,
            )

        assert isinstance(data_model_views_creator.data_model_views, DataModelViews)

    def test_create_data_model_views_contains_only_workers_with_sufficient_data(
        self, data_model_views_creator_init_params
    ):
        # Instantiate some local worker mocks
        local_worker_mocks_sufficient_data = [
            MagicMock(LocalWorker) for number_of_workers in range(5)
        ]
        local_worker_mocks_insufficient_data = [
            MagicMock(LocalWorker) for number_of_workers in range(5)
        ]

        # some of them with sufficient data
        for worker in local_worker_mocks_sufficient_data:
            worker.worker_id = "sufficientdataworker"
            table_info = self.TableInfoMock()
            table_info.schema_ = "dummy_schema"
            worker.create_data_model_views.return_value = [table_info]
        # and some of them without sufficient data
        for worker in local_worker_mocks_insufficient_data:
            worker.worker_id = "insufficientdataworker"
            worker.create_data_model_views.side_effect = InsufficientDataError("")

        data_model_views_creator = DataModelViewsCreator(
            local_workers=(
                local_worker_mocks_sufficient_data
                + local_worker_mocks_insufficient_data
            ),
            variable_groups=data_model_views_creator_init_params.variable_groups,
            var_filters=data_model_views_creator_init_params.var_filters,
            dropna=data_model_views_creator_init_params.dropna,
            check_min_rows=data_model_views_creator_init_params.check_min_rows,
            command_id=data_model_views_creator_init_params.command_id,
        )

        data_model_views_creator.create_data_model_views()

        # check that the data model views contains only workers with sufficient data
        data_model_views = data_model_views_creator.data_model_views.to_list()[0]
        workers = list(data_model_views.workers_tables_info.keys())
        assert set(workers) == set(local_worker_mocks_sufficient_data)

    def test_create_data_model_views_raises_error_when_all_workers_insufficient_data(
        self, data_model_views_creator_init_params
    ):
        # Instantiate local worker mocks, all of them without sufficient data
        local_worker_mocks = [MagicMock(LocalWorker) for number_of_workers in range(10)]
        for worker_mock in local_worker_mocks:
            worker_mock.worker_id = "some_id.."
            worker_mock.create_data_model_views.side_effect = InsufficientDataError("")

        data_model_views_creator = DataModelViewsCreator(
            local_workers=local_worker_mocks,
            variable_groups=data_model_views_creator_init_params.variable_groups,
            var_filters=data_model_views_creator_init_params.var_filters,
            dropna=data_model_views_creator_init_params.dropna,
            check_min_rows=data_model_views_creator_init_params.check_min_rows,
            command_id=data_model_views_creator_init_params.command_id,
        )
        with pytest.raises(InsufficientDataError):
            data_model_views_creator.create_data_model_views()


class AsyncResult:
    pass

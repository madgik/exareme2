from unittest.mock import patch

import pytest

from exareme2.controller import logger as ctrl_logger
from exareme2.controller.services.exareme2.algorithm_flow_engine_interface import (
    Exareme2AlgorithmFlowEngineInterface,
)
from exareme2.controller.services.exareme2.algorithm_flow_engine_interface import (
    InitializationParams,
)
from exareme2.controller.services.exareme2.algorithm_flow_engine_interface import (
    SMPCParams,
)
from exareme2.smpc_cluster_communication import DifferentialPrivacyParams


class TestExareme2AlgorithmFlowEngineInterface:
    @pytest.fixture
    def algorithm_execution_engine_init_params(self):
        return InitializationParams(
            smpc_params=SMPCParams(
                smpc_enabled=True,
                smpc_optional=False,
                dp_params=DifferentialPrivacyParams(sensitivity=1, privacy_budget=1),
            ),
            request_id="dummyrequestid",
            algo_flags="dumyalgoflags",
        )

    @pytest.fixture
    def algorithm_execution_engine(self, algorithm_execution_engine_init_params):
        return Exareme2AlgorithmFlowEngineInterface(
            initialization_params=algorithm_execution_engine_init_params,
            command_id_generator="dummy_id_generator",
            workers="dummy_workers",
        )

    def test_initialiazation(self, algorithm_execution_engine_init_params):
        dummy_command_id_generator = "dummy_id_generator"
        dummy_workers = "dummy_workers"

        algorithm_execution_engine = Exareme2AlgorithmFlowEngineInterface(
            initialization_params=algorithm_execution_engine_init_params,
            command_id_generator=dummy_command_id_generator,
            workers=dummy_workers,
        )

        assert algorithm_execution_engine._logger == ctrl_logger.get_request_logger(
            request_id=algorithm_execution_engine_init_params.request_id
        )
        assert (
            algorithm_execution_engine._algorithm_execution_flags
            == algorithm_execution_engine_init_params.algo_flags
        )
        assert (
            algorithm_execution_engine._smpc_params
            == algorithm_execution_engine_init_params.smpc_params
        )
        assert (
            algorithm_execution_engine._command_id_generator
            == dummy_command_id_generator
        )
        assert algorithm_execution_engine._workers == dummy_workers

    # NOTE: This unittest was written during the 'differential privacy' feature implementation. The
    # only thing it actually tests is that the _share_local_smpc_tables_to_global method passes the
    # correct/expected arguments to the function related to the 'differential privacy' mechanism it
    # subsequently calls, the trigger_smpc_operations function, no other checks are taking place.
    # The _share_local_smpc_tables_to_global method is anyway too lengthy to be fully unittested and
    # it would be much easier if refactored to smaller methods/functions.
    def test_share_local_smpc_tables_to_global(self, algorithm_execution_engine):
        with patch(
            "exareme2.controller.services.exareme2.algorithm_flow_engine_interface.Exareme2AlgorithmFlowEngineInterface._share_local_table_to_global"
        ) as mock_share_local_table_to_global, patch(
            "exareme2.controller.services.exareme2.algorithm_flow_engine_interface.load_data_to_smpc_clients",
        ) as mock_load_data_to_smpc_clients, patch(
            "exareme2.controller.services.exareme2.algorithm_flow_engine_interface.trigger_smpc_operations"
        ) as mock_trigger_smpc_operations, patch(
            "exareme2.controller.services.exareme2.algorithm_flow_engine_interface.wait_for_smpc_results_to_be_ready"
        ) as mock_wait_for_smpc_results_to_be_ready, patch(
            "exareme2.controller.services.exareme2.algorithm_flow_engine_interface.get_smpc_results"
        ) as mock_get_smpc_results, patch(
            "exareme2.controller.services.exareme2.algorithm_flow_engine_interface.GlobalWorkerSMPCTables"
        ) as MockGlobalWorkerSMPCTables, patch(
            "exareme2.controller.services.exareme2.algorithm_flow_engine_interface.SMPCTablesInfo"
        ) as MockSMPCTablesInfo:
            command_id = 12345

            mock_load_data_to_smpc_clients_return_value = "a_dummy_value"
            mock_load_data_to_smpc_clients.return_value = (
                mock_load_data_to_smpc_clients_return_value
            )

            mock_trigger_smpc_operations.return_value = [
                "another_dummy_value",
                "another_dummy_value",
                "another_dummy_value",
            ]

            mock_get_smpc_results.return_value = [
                "another_dummy_value",
                "another_dummy_value",
                "another_dummy_value",
            ]

            class MockLocalWorkersSMPCTables:
                template_local_workers_table = ""

            class MockWorkers:
                class MockGlobalWorker:
                    context_id = "contextid"

                    def validate_smpc_templates_match(self, arg):
                        pass

                global_worker = MockGlobalWorker()

            algorithm_execution_engine._workers = MockWorkers()

            algorithm_execution_engine._share_local_smpc_tables_to_global(
                local_workers_smpc_tables=MockLocalWorkersSMPCTables(),
                command_id=command_id,
            )

            mock_trigger_smpc_operations.assert_called_once_with(
                logger=algorithm_execution_engine._logger,
                context_id=algorithm_execution_engine._workers.global_worker.context_id,
                command_id=command_id,
                smpc_clients_per_op=mock_load_data_to_smpc_clients_return_value,
                dp_params=algorithm_execution_engine._smpc_params.dp_params,
            )

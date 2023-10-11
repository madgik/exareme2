from typing import Dict
from typing import List
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from exareme2.controller import controller_logger as ctrl_logger
from exareme2.controller.algorithm_execution_engine import AlgorithmExecutionEngine
from exareme2.controller.algorithm_execution_engine import InitializationParams
from exareme2.controller.algorithm_execution_engine import SMPCParams
from exareme2.controller.nodes import _INode
from exareme2.node_tasks_DTOs import ColumnInfo
from exareme2.node_tasks_DTOs import DType
from exareme2.node_tasks_DTOs import TableData
from exareme2.node_tasks_DTOs import TableInfo
from exareme2.node_tasks_DTOs import TableSchema
from exareme2.smpc_DTOs import DifferentialPrivacyParams


class TestAlgorithmExecutionEngine:
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
        return AlgorithmExecutionEngine(
            initialization_params=algorithm_execution_engine_init_params,
            command_id_generator="dummy_id_generator",
            nodes="dummy_nodes",
        )

    def test_initialiazation(self, algorithm_execution_engine_init_params):
        dummy_command_id_generator = "dummy_id_generator"
        dummy_nodes = "dummy_nodes"

        algorithm_execution_engine = AlgorithmExecutionEngine(
            initialization_params=algorithm_execution_engine_init_params,
            command_id_generator=dummy_command_id_generator,
            nodes=dummy_nodes,
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
        assert algorithm_execution_engine._nodes == dummy_nodes

    # NOTE: This unittest was written during the 'differential privacy' feature implementation. The
    # only thing it actually tests is that the _share_local_smpc_tables_to_global method passes the
    # correct/expected arguments to the function related to the 'differential privacy' mechanism it
    # subsequently calls, the trigger_smpc_operations function, no other checks are taking place.
    # The _share_local_smpc_tables_to_global method is anyway too lengthy to be fully unittested and
    # it would be much easier if refactored to smaller methods/functions.
    def test_share_local_smpc_tables_to_global(self, algorithm_execution_engine):
        with patch(
            "exareme2.controller.algorithm_execution_engine.AlgorithmExecutionEngine._share_local_table_to_global"
        ) as mock_share_local_table_to_global, patch(
            "exareme2.controller.algorithm_execution_engine.load_data_to_smpc_clients",
        ) as mock_load_data_to_smpc_clients, patch(
            "exareme2.controller.algorithm_execution_engine.trigger_smpc_operations"
        ) as mock_trigger_smpc_operations, patch(
            "exareme2.controller.algorithm_execution_engine.wait_for_smpc_results_to_be_ready"
        ) as mock_wait_for_smpc_results_to_be_ready, patch(
            "exareme2.controller.algorithm_execution_engine.get_smpc_results"
        ) as mock_get_smpc_results, patch(
            "exareme2.controller.algorithm_execution_engine.GlobalNodeSMPCTables"
        ) as MockGlobalNodeSMPCTables, patch(
            "exareme2.controller.algorithm_execution_engine.SMPCTablesInfo"
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

            class MockLocalNodesSMPCTables:
                template_local_nodes_table = ""

            class MockNodes:
                class MockGlobalNode:
                    context_id = "contextid"

                    def validate_smpc_templates_match(self, arg):
                        pass

                global_node = MockGlobalNode()

            algorithm_execution_engine._nodes = MockNodes()

            algorithm_execution_engine._share_local_smpc_tables_to_global(
                local_nodes_smpc_tables=MockLocalNodesSMPCTables(),
                command_id=command_id,
            )

            mock_trigger_smpc_operations.assert_called_once_with(
                logger=algorithm_execution_engine._logger,
                context_id=algorithm_execution_engine._nodes.global_node.context_id,
                command_id=command_id,
                smpc_clients_per_op=mock_load_data_to_smpc_clients_return_value,
                dp_params=algorithm_execution_engine._smpc_params.dp_params,
            )

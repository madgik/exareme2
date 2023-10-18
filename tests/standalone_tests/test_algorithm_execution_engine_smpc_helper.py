import unittest.mock
from logging import Logger

from exareme2.controller.services.in_database.smpc_cluster_comm_helpers import (
    _trigger_smpc_operation,
)
from exareme2.controller.services.in_database.smpc_cluster_comm_helpers import (
    get_smpc_job_id,
)
from exareme2.controller.services.in_database.smpc_cluster_comm_helpers import (
    trigger_smpc_operations,
)
from exareme2.smpc_cluster_communication import DifferentialPrivacyParams
from exareme2.smpc_cluster_communication import SMPCRequestType
from exareme2.smpc_cluster_communication import create_payload
from exareme2.utils import AttrDict


def test_trigger_smpc_operation():
    logger = Logger("dummy_logger")
    context_id = "contextid"
    command_id = 0
    op_type = SMPCRequestType.INT_SUM
    smpc_op_clients = ["client1", "client2", "client3"]
    dp_params = DifferentialPrivacyParams(
        sensitivity=1,
        privacy_budget=1,
    )

    with unittest.mock.patch(
        "exareme2.controller.services.in_database.smpc_cluster_comm_helpers.trigger_smpc"
    ) as mock_trigger_smpc, unittest.mock.patch(
        "exareme2.controller.services.in_database.smpc_cluster_comm_helpers.ctrl_config"
    ) as mock_ctrl_config:
        mock_ctrl_config.smpc = AttrDict({"coordinator_address": "dummy_address"})

        return_val = _trigger_smpc_operation(
            logger=logger,
            context_id=context_id,
            command_id=command_id,
            op_type=op_type,
            smpc_op_clients=smpc_op_clients,
            dp_params=dp_params,
        )

        # assert call to trigger_smpc had the correct arguments
        mock_trigger_smpc.assert_called_once_with(
            logger=logger,
            coordinator_address=mock_ctrl_config.smpc["coordinator_address"],
            jobid=get_smpc_job_id(
                context_id=context_id,
                command_id=command_id,
                operation=op_type,
            ),
            payload=create_payload(
                computation_type=op_type, clients=smpc_op_clients, dp_params=dp_params
            ),
        )

        # assert it returns the expected values
        # when smpc_op_clients argument not empty
        assert return_val == True

        # and when smpc_op_clients argument empty
        return_val = _trigger_smpc_operation(
            logger=logger,
            context_id=context_id,
            command_id=command_id,
            op_type=op_type,
            smpc_op_clients=[],
        )
        assert return_val == False


def test_trigger_smpc_operations():
    logger = Logger("dummy_logger")
    context_id = "contextid"
    command_id = 0
    op_type = SMPCRequestType.INT_SUM
    smpc_op_clients = ["client1", "client2", "client3"]
    dp_params = DifferentialPrivacyParams(
        sensitivity=1,
        privacy_budget=1,
    )

    with unittest.mock.patch(
        "exareme2.controller.services.in_database.smpc_cluster_comm_helpers._trigger_smpc_operation"
    ) as mock_trigger_smpc_operation:
        expected_return1 = 123
        expected_return2 = 456
        expected_return3 = 789
        expected = [expected_return1, expected_return2, expected_return3]

        def side_effect(*args, **kwargs):
            return expected.pop(0)

        mock_trigger_smpc_operation.side_effect = side_effect

        returned = trigger_smpc_operations(
            logger=logger,
            context_id=context_id,
            command_id=command_id,
            smpc_clients_per_op=smpc_op_clients,
            dp_params=dp_params,
        )

        # check tringger_smpc_operations returns a tuple of the 3 values returned
        # by _trigger_smpc_operation
        assert returned == (expected_return1, expected_return2, expected_return3)

        # check _trigger_smpc_operation was called exactly 3 times
        assert mock_trigger_smpc_operation.call_count == 3

        # check it was called with the correct args
        expected_call_args = [
            [
                logger,
                context_id,
                command_id,
                SMPCRequestType.SUM,
                smpc_op_clients[0],
                dp_params,
            ],
            [
                logger,
                context_id,
                command_id,
                SMPCRequestType.MIN,
                smpc_op_clients[1],
                dp_params,
            ],
            [
                logger,
                context_id,
                command_id,
                SMPCRequestType.MAX,
                smpc_op_clients[2],
                dp_params,
            ],
        ]
        called_args_list = [
            list(call.args) for call in mock_trigger_smpc_operation.call_args_list
        ]
        for args in expected_call_args:
            assert args in called_args_list

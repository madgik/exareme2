from unittest.mock import patch

import pytest

from mipengine import AttrDict
from mipengine.node.tasks.udfs import _validate_smpc_usage
from mipengine.node_exceptions import SMPCUsageError


def get_validate_smpc_usage_success_cases():
    validate_smpc_usage_success_cases = [
        (
            {"smpc": {"enabled": True, "optional": False}},
            True,
        ),
        (
            {
                "smpc": {
                    "enabled": False,
                }
            },
            False,
        ),
        (
            {"smpc": {"enabled": True, "optional": True}},
            False,
        ),
    ]
    return validate_smpc_usage_success_cases


@pytest.mark.parametrize(
    "node_config, use_smpc", get_validate_smpc_usage_success_cases()
)
def test_validate_smpc_usage_success_cases(node_config, use_smpc):
    with patch(
        "mipengine.node.tasks.udfs.node_config",
        AttrDict(node_config),
    ):
        try:
            _validate_smpc_usage(use_smpc)
        except Exception as exc:
            pytest.fail(f"No exception should be raised. Exception: {exc}")


def get_validate_smpc_usage_fail_cases():
    validate_smpc_usage_fail_cases = [
        (
            {"smpc": {"enabled": True, "optional": False}},
            False,
            (
                SMPCUsageError,
                "The computation cannot be made without SMPC. SMPC usage is not optional.",
            ),
        ),
        (
            {
                "smpc": {
                    "enabled": False,
                }
            },
            True,
            (
                SMPCUsageError,
                "SMPC cannot be used, since it's not enabled on the node.",
            ),
        ),
    ]
    return validate_smpc_usage_fail_cases


@pytest.mark.parametrize(
    "node_config, use_smpc, exception", get_validate_smpc_usage_fail_cases()
)
def test_validate_smpc_usage_fail_cases(node_config, use_smpc, exception):
    with patch(
        "mipengine.node.tasks.udfs.node_config",
        AttrDict(node_config),
    ):
        exception_type, exception_message = exception
        with pytest.raises(exception_type, match=exception_message):
            _validate_smpc_usage(use_smpc)

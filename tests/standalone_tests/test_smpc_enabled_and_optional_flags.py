import pytest

from mipengine.smpc_cluster_comm_helpers import SMPCUsageError
from mipengine.smpc_cluster_comm_helpers import validate_smpc_usage


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
                    "optional": False,
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


@pytest.mark.smpc
@pytest.mark.parametrize(
    "node_config, use_smpc", get_validate_smpc_usage_success_cases()
)
def test_validate_smpc_usage_success_cases(node_config, use_smpc):
    try:
        validate_smpc_usage(
            use_smpc, node_config["smpc"]["enabled"], node_config["smpc"]["optional"]
        )
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
                    "optional": False,
                }
            },
            True,
            (
                SMPCUsageError,
                "SMPC cannot be used, since it's not enabled.",
            ),
        ),
    ]
    return validate_smpc_usage_fail_cases


@pytest.mark.smpc
@pytest.mark.parametrize(
    "node_config, use_smpc, exception", get_validate_smpc_usage_fail_cases()
)
def test_validate_smpc_usage_fail_cases(node_config, use_smpc, exception):
    exception_type, exception_message = exception
    with pytest.raises(exception_type, match=exception_message):
        validate_smpc_usage(
            use_smpc, node_config["smpc"]["enabled"], node_config["smpc"]["optional"]
        )

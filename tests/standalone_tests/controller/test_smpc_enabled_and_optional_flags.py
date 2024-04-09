import pytest

from exareme2.smpc_cluster_communication import DifferentialPrivacyParams
from exareme2.smpc_cluster_communication import DPRequestData
from exareme2.smpc_cluster_communication import SMPCRequestData
from exareme2.smpc_cluster_communication import SMPCRequestType
from exareme2.smpc_cluster_communication import SMPCUsageError
from exareme2.smpc_cluster_communication import create_payload
from exareme2.smpc_cluster_communication import validate_smpc_usage


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
    "worker_config, use_smpc", get_validate_smpc_usage_success_cases()
)
def test_validate_smpc_usage_success_cases(worker_config, use_smpc):
    try:
        validate_smpc_usage(
            use_smpc,
            worker_config["smpc"]["enabled"],
            worker_config["smpc"]["optional"],
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
    "worker_config, use_smpc, exception", get_validate_smpc_usage_fail_cases()
)
def test_validate_smpc_usage_fail_cases(worker_config, use_smpc, exception):
    exception_type, exception_message = exception
    with pytest.raises(exception_type, match=exception_message):
        validate_smpc_usage(
            use_smpc,
            worker_config["smpc"]["enabled"],
            worker_config["smpc"]["optional"],
        )


def test_create_payload():
    computation_type = SMPCRequestType.INT_SUM
    clients = ["client1", "client2", "client3"]
    dp_sensitivity = 1
    dp_privacy_budget = 1

    # test without differential privacy parameters
    expected_without_pd = SMPCRequestData(
        computationType=computation_type,
        clients=clients,
    )

    assert expected_without_pd == create_payload(
        computation_type=computation_type,
        clients=clients,
    )

    # test with differential privacy parameters
    expected_with_pd = SMPCRequestData(
        computationType=computation_type,
        clients=clients,
        dp=DPRequestData(
            c=dp_sensitivity,
            e=dp_privacy_budget,
        ),
    )

    assert expected_with_pd == create_payload(
        computation_type=computation_type,
        clients=clients,
        dp_params=DifferentialPrivacyParams(
            sensitivity=dp_sensitivity, privacy_budget=dp_privacy_budget
        ),
    )

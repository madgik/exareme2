from unittest.mock import Mock
from unittest.mock import patch

import pytest

from exareme2 import AttrDict
from exareme2.controller.api.endpoint import create_controller
from exareme2.smpc_cluster_communication import DifferentialPrivacyParams


@pytest.fixture
def controller_config_1():
    controller_config = AttrDict({})
    controller_config.rabbitmq = AttrDict(
        {
            "celery_tasks_timeout": 5,
            "celery_run_udf_task_timeout": 10,
        }
    )
    controller_config.smpc = AttrDict({"enabled": False, "optional": False})
    b
    controller_config.smpc.dp = AttrDict(
        {"enabled": True, "sensitivity": 1, "privacy_budget": 1}
    )
    return controller_config


def test_create_controller():
    with patch(
        "exareme2.controller.api.endpoint.controller_config"
    ) as mock_controller_config:
        # NOTE the following setup of the controller_config mock should reside in a fixture.
        # Nevertheless, it seems there is some complex behavior in the endpoint.py module that is
        # somehow affecting the patching process in a way moving the setup of the mock in a fixture
        # does not work for me...

        # mock 1
        mock_controller_config.rabbitmq = AttrDict(
            {
                "celery_tasks_timeout": 5,
                "celery_run_udf_task_timeout": 10,
            }
        )
        mock_controller_config.smpc = AttrDict({"enabled": True, "optional": False})
        mock_controller_config.smpc.dp = AttrDict(
            {"enabled": True, "sensitivity": 1, "privacy_budget": 1}
        )

        mock_cleaner = Mock()
        mock_node_landscape_aggregator = Mock()
        controller = create_controller(
            node_landscape_aggregator=mock_node_landscape_aggregator,
            cleaner=mock_cleaner,
        )
        assert (
            controller._celery_tasks_timeout
            == mock_controller_config.rabbitmq.celery_tasks_timeout
        )
        assert (
            controller._celery_run_udf_task_timeout
            == mock_controller_config.rabbitmq.celery_run_udf_task_timeout
        )
        assert (
            controller._smpc_params.smpc_enabled == mock_controller_config.smpc.enabled
        )
        assert (
            controller._smpc_params.smpc_optional
            == mock_controller_config.smpc.optional
        )
        assert controller._smpc_params.dp_params == DifferentialPrivacyParams(
            sensitivity=mock_controller_config.dp.sensitivity,
            privacy_budget=mock_controller_config.dp.privacy_budget,
        )

        assert controller._node_landscape_aggregator == mock_node_landscape_aggregator
        assert controller._cleaner == mock_cleaner

        # mock 2
        mock_controller_config.rabbitmq = AttrDict(
            {
                "celery_tasks_timeout": 123,
                "celery_run_udf_task_timeout": 456,
            }
        )
        mock_controller_config.smpc = AttrDict({"enabled": True, "optional": True})
        mock_controller_config.smpc.dp = AttrDict({"enabled": False})

        controller = create_controller(
            node_landscape_aggregator=mock_node_landscape_aggregator,
            cleaner=mock_cleaner,
        )
        assert (
            controller._celery_tasks_timeout
            == mock_controller_config.rabbitmq.celery_tasks_timeout
        )
        assert (
            controller._celery_run_udf_task_timeout
            == mock_controller_config.rabbitmq.celery_run_udf_task_timeout
        )
        assert (
            controller._smpc_params.smpc_enabled == mock_controller_config.smpc.enabled
        )
        assert (
            controller._smpc_params.smpc_optional
            == mock_controller_config.smpc.optional
        )
        assert controller._smpc_params.dp_params == None

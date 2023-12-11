import requests

from tests.prod_env_tests import healthcheck_url


def test_healthcheck():
    response = requests.get(healthcheck_url)
    assert response.status_code == 200

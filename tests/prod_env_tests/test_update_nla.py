import requests

from tests.prod_env_tests import wla_url


def test_update_wla():
    response = requests.post(wla_url)
    assert response.status_code == 200

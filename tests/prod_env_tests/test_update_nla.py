import requests

from tests.prod_env_tests import nla_url


def test_update_nla():
    response = requests.post(nla_url)
    assert response.status_code == 200

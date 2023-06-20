import json

import requests

from tests.prod_env_tests import transformers_url


def test_get_transformers():
    request = requests.get(transformers_url)
    result = json.loads(request.text)
    assert len(result) > 0

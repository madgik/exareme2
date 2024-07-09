import json
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

from pydantic import ValidationError

from exareme2.algorithms.flower.inputdata_preprocessing import HEADERS
from exareme2.algorithms.flower.inputdata_preprocessing import RESULT_URL
from exareme2.algorithms.flower.inputdata_preprocessing import error_handling
from exareme2.algorithms.flower.inputdata_preprocessing import get_enumerations
from exareme2.algorithms.flower.inputdata_preprocessing import get_input
from exareme2.algorithms.flower.inputdata_preprocessing import post_result


class TestAPIMethods(unittest.TestCase):
    @patch("requests.post")
    def test_error_handling(self, mock_post):
        error = Exception("Test error")
        error_handling(error)
        self.assertTrue(mock_post.called)
        self.assertEqual(mock_post.call_args[1]["headers"], HEADERS)
        error_data = json.loads(mock_post.call_args[1]["data"])
        self.assertIn("error", error_data)
        self.assertEqual(error_data["error"], "Test error")

    @patch("requests.post")
    def test_post_result_success(self, mock_post):
        mock_post.return_value.status_code = 200
        result = {"key": "value"}
        post_result(result)
        self.assertTrue(mock_post.called)
        self.assertEqual(mock_post.call_args[1]["headers"], HEADERS)
        self.assertEqual(mock_post.call_args[0][0], RESULT_URL)
        result_data = json.loads(mock_post.call_args[1]["data"])
        self.assertEqual(result_data, result)

    @patch("requests.post")
    def test_post_result_failure(self, mock_post):
        mock_post.side_effect = [
            MagicMock(status_code=500, text="Internal Server Error"),
            MagicMock(status_code=200),
        ]
        result = {"key": "value"}
        post_result(result)
        self.assertEqual(mock_post.call_count, 2)
        error_call = mock_post.call_args_list[1]
        error_data = json.loads(error_call[1]["data"])
        self.assertIn("error", error_data)
        self.assertIn("Internal Server Error", error_data["error"])

    @patch("requests.get")
    @patch("requests.post")
    def test_get_input_success(self, mock_post, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = json.dumps(
            {
                "data_model": "model",
                "datasets": ["dataset1"],
                "filters": None,
                "y": ["target"],
                "x": ["feature1"],
            }
        )
        input_data = get_input()
        self.assertTrue(mock_get.called)
        self.assertEqual(input_data.data_model, "model")
        self.assertEqual(input_data.datasets, ["dataset1"])
        self.assertEqual(input_data.y, ["target"])
        self.assertEqual(input_data.x, ["feature1"])

    @patch("requests.get")
    @patch("requests.post")
    def test_get_input_failure(self, mock_post, mock_get):
        mock_get.return_value.status_code = 500
        mock_get.return_value.text = "Internal Server Error"
        with self.assertRaises(ValidationError):
            get_input()
        self.assertTrue(mock_get.called)
        self.assertTrue(mock_post.called)
        error_call = mock_post.call_args_list[0]
        error_data = json.loads(error_call[1]["data"])
        self.assertIn("error", error_data)
        self.assertIn("Internal Server Error", error_data["error"])

    @patch("requests.get")
    @patch("requests.post")
    def test_get_enumerations_success(self, mock_post, mock_get):
        cdes_metadata = {
            "model": {
                "variable": {"enumerations": {"code1": "label1", "code2": "label2"}}
            }
        }
        mock_get.return_value.status_code = 200
        mock_get.return_value.json = MagicMock(return_value=cdes_metadata)
        enumerations = get_enumerations("model", "variable")
        self.assertTrue(mock_get.called)
        self.assertEqual(enumerations, ["code1", "code2"])

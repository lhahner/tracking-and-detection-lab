import unittest

from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

class TestInferenceRoute(unittest.TestCase):
    def test_create_model(self):
        payload = {
            "detector": "yolo",
            "tracker": "SORT",
            "pre_processing": [],
            "post_processing": [],
            "tracker": "",
            "dataset": ""
            }
        response_expected = {"status": "success"}
        response = client.post("/inference/run/", json=payload)
        data = response.json()
        self.assertEqual(response_expected, response)
        
    def test_available_resources_endpoint(self):
        response = client.get("/inference/available")
        

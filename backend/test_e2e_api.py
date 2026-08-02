"""
Automated E2E API integration tests for Rail-Kick endpoints (POST /analyze).
Tests full pipeline: HTTP upload -> felt detection -> table warp -> ball detection -> shot calculation -> base64 image generation.
"""
import io
import os
import unittest
from fastapi.testclient import TestClient
from main import app


class TestE2EAPIIntegration(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        self.base_dir = os.path.dirname(__file__)
        self.root_dir = os.path.dirname(self.base_dir)

    def test_e2e_analyze_example_jpg(self):
        img_path = os.path.join(self.root_dir, "example.jpg")
        if not os.path.exists(img_path):
            self.skipTest("example.jpg not found in workspace root")

        with open(img_path, "rb") as f:
            response = self.client.post(
                "/analyze",
                files={"image": ("example.jpg", f, "image/jpeg")},
                data={"felt_color": "auto"},
            )

        self.assertEqual(response.status_code, 200, f"API error: {response.text}")
        data = response.json()

        # Validate response schema
        self.assertIn("table_dims_mm", data)
        self.assertEqual(data["table_dims_mm"]["width"], 2540.0)
        self.assertEqual(data["table_dims_mm"]["height"], 1270.0)

        self.assertEqual(len(data["pockets"]), 6)
        self.assertEqual(len(data["diamonds"]), 18)
        self.assertGreaterEqual(len(data["balls"]), 3)

        self.assertIn("direct_shots", data)
        self.assertIn("bank_shots", data)
        self.assertIn("kick_shots", data)

        self.assertTrue(len(data["annotated_image_b64"]) > 100)
        self.assertTrue(len(data["cv_diagram_b64"]) > 100)

    def test_e2e_analyze_example_1_jpg(self):
        img_path = os.path.join(self.root_dir, "example_1.jpg")
        if not os.path.exists(img_path):
            self.skipTest("example_1.jpg not found in workspace root")

        with open(img_path, "rb") as f:
            response = self.client.post(
                "/analyze",
                files={"image": ("example_1.jpg", f, "image/jpeg")},
                data={"felt_color": "auto"},
            )

        self.assertEqual(response.status_code, 200, f"API error: {response.text}")
        data = response.json()

        self.assertEqual(len(data["pockets"]), 6)
        self.assertEqual(len(data["diamonds"]), 18)
        self.assertGreaterEqual(len(data["balls"]), 3)
        self.assertTrue(len(data["annotated_image_b64"]) > 100)
        self.assertTrue(len(data["cv_diagram_b64"]) > 100)

    def test_e2e_analyze_simonis_blue_table_jpg(self):
        img_path = os.path.join(self.base_dir, "fixtures", "simonis_blue_table.jpg")
        if not os.path.exists(img_path):
            self.skipTest("simonis_blue_table.jpg not found in fixtures")

        with open(img_path, "rb") as f:
            response = self.client.post(
                "/analyze",
                files={"image": ("simonis_blue_table.jpg", f, "image/jpeg")},
                data={"felt_color": "blue"},
            )

        self.assertEqual(response.status_code, 200, f"API error: {response.text}")
        data = response.json()

        self.assertEqual(len(data["pockets"]), 6)
        self.assertEqual(len(data["diamonds"]), 18)
        self.assertGreaterEqual(len(data["balls"]), 3)
        self.assertTrue(len(data["annotated_image_b64"]) > 100)
        self.assertTrue(len(data["cv_diagram_b64"]) > 100)


if __name__ == "__main__":
    unittest.main()

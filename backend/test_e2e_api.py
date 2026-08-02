"""E2E API tests with deterministic custom-YOLO predictions."""
import io
import unittest

from fastapi.testclient import TestClient
from PIL import Image

from main import app
from test_support import FakePoolDetector


def camera_jpeg() -> bytes:
    image = Image.new("RGB", (1400, 900), (20, 110, 150))
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=90)
    return output.getvalue()


class TestE2EAPIIntegration(unittest.TestCase):
    def setUp(self):
        app.state.cv_detector = FakePoolDetector()
        app.state.cv_model_error = None
        self.client = TestClient(app)

    def test_upload_through_yolo_to_shot_response(self):
        response = self.client.post(
            "/analyze",
            files={"image": ("iphone16.jpg", camera_jpeg(), "image/jpeg")},
        )

        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(data["table_dims_mm"], {"width": 2540.0, "height": 1270.0})
        self.assertEqual(len(data["pockets"]), 6)
        self.assertEqual(len(data["diamonds"]), 18)
        self.assertEqual(len(data["balls"]), 3)
        self.assertTrue(data["cue_detected"])
        self.assertIn("direct_shots", data)
        self.assertIn("bank_shots", data)
        self.assertIn("kick_shots", data)
        self.assertGreater(len(data["annotated_image_b64"]), 100)
        self.assertGreater(len(data["cv_diagram_b64"]), 100)

    def test_heic_content_type_is_accepted(self):
        # Decoder selection is content-based, so JPEG bytes also exercise the HEIC API branch.
        response = self.client.post(
            "/analyze",
            files={"image": ("iphone16.heic", camera_jpeg(), "image/heic")},
        )
        self.assertEqual(response.status_code, 200, response.text)

    def test_invalid_image_returns_400(self):
        response = self.client.post(
            "/analyze",
            files={"image": ("broken.jpg", b"not an image", "image/jpeg")},
        )
        self.assertEqual(response.status_code, 400)

    def test_yolo_table_miss_returns_422(self):
        app.state.cv_detector = FakePoolDetector(include_table=False)
        response = self.client.post(
            "/analyze",
            files={"image": ("iphone16.jpg", camera_jpeg(), "image/jpeg")},
        )
        self.assertEqual(response.status_code, 422)

    def test_unavailable_custom_weights_returns_503(self):
        app.state.cv_detector = None
        app.state.cv_model_error = "Custom YOLOv8s weights are unavailable."
        response = self.client.post(
            "/analyze",
            files={"image": ("iphone16.jpg", camera_jpeg(), "image/jpeg")},
        )
        self.assertEqual(response.status_code, 503)
        self.assertIn("weights", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()

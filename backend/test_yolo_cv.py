import unittest
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from cv_module import (
    ModelUnavailableError,
    YoloV8SmallDetector,
    analyse_image,
    select_inference_device,
)
from test_support import FakePoolDetector


class _Availability:
    def __init__(self, available):
        self._available = available

    def is_available(self):
        return self._available


def fake_torch(cuda=False, mps=False):
    return SimpleNamespace(
        cuda=_Availability(cuda),
        backends=SimpleNamespace(mps=_Availability(mps)),
    )


class TestDeviceSelection(unittest.TestCase):
    def test_missing_custom_checkpoint_fails_without_downloading(self):
        with self.assertRaises(ModelUnavailableError):
            YoloV8SmallDetector("/definitely/missing/rail-kick-model.pt")

    def test_auto_prefers_nvidia_cuda(self):
        self.assertEqual(select_inference_device(fake_torch(True, True)), "cuda:0")

    def test_auto_uses_mps_on_mac_without_cuda(self):
        self.assertEqual(select_inference_device(fake_torch(False, True)), "mps")

    def test_auto_falls_back_to_cpu(self):
        self.assertEqual(select_inference_device(fake_torch()), "cpu")

    def test_explicit_unavailable_gpu_fails_clearly(self):
        with self.assertRaises(ModelUnavailableError):
            select_inference_device(fake_torch(), requested="cuda")


class _Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class _FakeUltralyticsModel:
    names = {0: "table", 1: "cue_ball"}

    def predict(self, **kwargs):
        boxes = SimpleNamespace(
            xyxy=_Tensor([[10, 20, 300, 200], [100, 100, 140, 140]]),
            cls=_Tensor([0, 1]),
            conf=_Tensor([0.99, 0.95]),
        )
        masks = SimpleNamespace(
            xy=[
                np.array([[10, 20], [300, 20], [300, 200], [10, 200]]),
                np.array([[100, 100], [140, 100], [140, 140], [100, 140]]),
            ]
        )
        return [SimpleNamespace(boxes=boxes, masks=masks, names=self.names)]


class TestUltralyticsAdapter(unittest.TestCase):
    def test_converts_ultralytics_results_to_stable_contract(self):
        with tempfile.NamedTemporaryFile(suffix=".pt") as weights:
            fake_module = SimpleNamespace(YOLO=lambda *args, **kwargs: _FakeUltralyticsModel())
            with patch.dict(sys.modules, {"ultralytics": fake_module}):
                detector = YoloV8SmallDetector(weights.name, device="cpu")
                detections = detector.predict(np.zeros((300, 400, 3), dtype=np.uint8))

        self.assertEqual([item.label for item in detections], ["table", "cue_ball"])
        self.assertEqual(detections[0].polygon.shape, (4, 2))


class TestYoloAnalysis(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((900, 1400, 3), dtype=np.uint8)

    def test_table_and_balls_are_converted_to_existing_contract(self):
        detector = FakePoolDetector()
        result = analyse_image(self.image, detector=detector)

        self.assertIsNotNone(result)
        self.assertEqual(detector.calls, 2)
        self.assertEqual(result["warped"].shape, (1280, 2560, 3))
        self.assertEqual(result["dims"].width, 2540.0)
        self.assertEqual(len(result["pockets"]), 6)
        self.assertEqual(len(result["diamonds"]), 18)
        self.assertEqual(len(result["balls"]), 3)
        self.assertTrue(result["cue_detected"])
        self.assertEqual(sum(ball.id == "cue" for ball in result["balls"]), 1)
        for ball in result["balls"]:
            self.assertGreaterEqual(ball.x, ball.radius_mm)
            self.assertGreaterEqual(ball.y, ball.radius_mm)

    def test_portrait_table_is_normalized_to_landscape(self):
        result = analyse_image(self.image, detector=FakePoolDetector(portrait=True))
        self.assertTrue(result["is_portrait"])
        self.assertEqual(result["warped"].shape, (1280, 2560, 3))

    def test_missing_table_returns_none(self):
        result = analyse_image(self.image, detector=FakePoolDetector(include_table=False))
        self.assertIsNone(result)

    def test_manual_cue_coordinates_are_preserved(self):
        result = analyse_image(
            self.image,
            detector=FakePoolDetector(),
            manual_cue_x=300.0,
            manual_cue_y=400.0,
        )
        cue = next(ball for ball in result["balls"] if ball.id == "cue")
        self.assertEqual((cue.x, cue.y), (300.0, 400.0))


if __name__ == "__main__":
    unittest.main()

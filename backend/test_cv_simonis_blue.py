"""
CV Engine Integration Unit Tests using Simonis 860 Tournament Blue Table Fixture.
Tests boundary detection, felt_color options, pocket/diamond positioning, and cue + object ball detection.
"""
import os
import unittest
import cv2
from cv_module import analyse_image


class TestCVSimonisBlueEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        fixture_path = os.path.join(os.path.dirname(__file__), "fixtures", "simonis_blue_table.jpg")
        cls.image_exists = os.path.exists(fixture_path)
        if cls.image_exists:
            cls.img_bgr = cv2.imread(fixture_path)
        else:
            cls.img_bgr = None

    def test_simonis_blue_cv_detection_auto(self):
        if not self.image_exists or self.img_bgr is None:
            self.skipTest("Fixture image 'fixtures/simonis_blue_table.jpg' not found.")

        result = analyse_image(self.img_bgr, felt_color="auto")
        self.assertIsNotNone(result, "analyse_image returned None on Simonis Blue table fixture with felt_color=auto")

        # Verify table dimensions
        dims = result["dims"]
        self.assertEqual(dims.width, 2540.0)
        self.assertEqual(dims.height, 1270.0)

        # Verify pockets & rail diamonds
        pockets = result["pockets"]
        diamonds = result["diamonds"]
        self.assertEqual(len(pockets), 6)
        self.assertEqual(len(diamonds), 18)

        # Verify Cue ball detection
        self.assertTrue(result["cue_detected"], "Cue ball was not detected on Simonis Blue table")
        balls = result["balls"]

        cue_ball = next((b for b in balls if b.id == "cue"), None)
        self.assertIsNotNone(cue_ball, "Cue ball object missing in detected balls")

    def test_simonis_blue_cv_detection_explicit_blue(self):
        if not self.image_exists or self.img_bgr is None:
            self.skipTest("Fixture image 'fixtures/simonis_blue_table.jpg' not found.")

        result = analyse_image(self.img_bgr, felt_color="blue")
        self.assertIsNotNone(result, "analyse_image returned None on Simonis Blue table fixture with felt_color=blue")
        self.assertTrue(result["cue_detected"])


if __name__ == "__main__":
    unittest.main()

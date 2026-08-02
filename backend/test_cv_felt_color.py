"""
Unit tests for user-selected felt color table detection logic in cv_module.py.
"""
import unittest
import numpy as np
import cv2
from cv_module import _detect_felt_contour, detect_table_and_warp


class TestCVFeltColorSelection(unittest.TestCase):

    def setUp(self):
        # Synthetic BGR image with a blue rectangle playfield in the center
        self.img = np.zeros((600, 800, 3), dtype=np.uint8)
        # Fill blue felt ROI (BGR: high blue ~ 200, HSV: H ~ 100)
        cv2.rectangle(self.img, (100, 100), (700, 500), (200, 100, 0), -1)

    def test_detect_felt_contour_blue_option(self):
        contour = _detect_felt_contour(self.img, felt_color="blue")
        self.assertIsNotNone(contour, "_detect_felt_contour failed to find blue felt with felt_color='blue'")

    def test_detect_felt_contour_green_option_fails_on_blue_image(self):
        contour = _detect_felt_contour(self.img, felt_color="green")
        self.assertIsNone(contour, "_detect_felt_contour should return None when felt_color='green' on a pure blue image")

    def test_detect_felt_contour_auto_option(self):
        contour = _detect_felt_contour(self.img, felt_color="auto")
        self.assertIsNotNone(contour, "_detect_felt_contour failed with felt_color='auto'")

    def test_detect_table_and_warp_with_felt_color(self):
        warped, H, dims, is_portrait = detect_table_and_warp(self.img, felt_color="blue")
        self.assertIsNotNone(warped)
        self.assertIsNotNone(H)


if __name__ == "__main__":
    unittest.main()

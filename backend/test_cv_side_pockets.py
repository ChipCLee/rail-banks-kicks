"""Orientation tests for YOLO table-mask corner normalization."""
import unittest

import numpy as np

from cv_module import analyse_image
from test_support import FakePoolDetector


class TestYoloTableOrientation(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((900, 1400, 3), dtype=np.uint8)

    def test_horizontal_long_rails_remain_landscape(self):
        result = analyse_image(self.image, detector=FakePoolDetector(portrait=False))
        self.assertFalse(result["is_portrait"])
        self.assertEqual(result["warped"].shape[:2], (1280, 2560))

    def test_vertical_long_rails_are_rotated_for_geometry(self):
        result = analyse_image(self.image, detector=FakePoolDetector(portrait=True))
        self.assertTrue(result["is_portrait"])
        self.assertEqual(result["warped"].shape[:2], (1280, 2560))


if __name__ == "__main__":
    unittest.main()

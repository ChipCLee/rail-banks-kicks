"""Compatibility tests for the retired felt-color API input."""
import unittest

import numpy as np

from cv_module import analyse_image
from test_support import FakePoolDetector


class TestFeltColorCompatibility(unittest.TestCase):
    def test_legacy_felt_color_does_not_change_learned_segmentation(self):
        image = np.zeros((600, 800, 3), dtype=np.uint8)
        auto = analyse_image(image, felt_color="auto", detector=FakePoolDetector())
        blue = analyse_image(image, felt_color="blue", detector=FakePoolDetector())
        self.assertEqual(auto["dims"], blue["dims"])
        self.assertEqual(auto["balls"], blue["balls"])


if __name__ == "__main__":
    unittest.main()

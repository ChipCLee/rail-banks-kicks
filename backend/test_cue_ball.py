"""
Unit tests for cue ball classification (including red-dot / measles cue ball).
"""
import unittest
import numpy as np
import cv2
from cv_module import get_ball_white_ratio, classify_ball


class TestCueBallClassification(unittest.TestCase):

    def test_pure_white_cue_ball(self):
        # Create a synthetic white ball image (100x100 pixels, BGR = 240, 240, 240)
        img = np.full((100, 100, 3), 240, dtype=np.uint8)
        label = classify_ball(img, 50, 50, 45)
        self.assertEqual(label, "cue")

    def test_measles_red_dot_cue_ball(self):
        # Create a synthetic white ball image with 6 small red dots
        img = np.full((100, 100, 3), 240, dtype=np.uint8)
        # Add 6 red dots (BGR = 0, 0, 220)
        dot_centers = [(30, 30), (70, 30), (50, 50), (30, 70), (70, 70), (50, 20)]
        for cx, cy in dot_centers:
            cv2.circle(img, (cx, cy), 5, (0, 0, 220), -1)

        # White ratio should still be high (> 80%)
        white_ratio = get_ball_white_ratio(img)
        self.assertGreater(white_ratio, 0.80)

        # Classification should recognize it as "cue"
        label = classify_ball(img, 50, 50, 45)
        self.assertEqual(label, "cue")


if __name__ == "__main__":
    unittest.main()

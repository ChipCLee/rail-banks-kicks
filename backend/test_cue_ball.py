"""
Unit tests for cue ball and solid/stripe ball classification.
"""
import unittest
import numpy as np
import cv2
from cv_module import get_ball_white_ratio, classify_ball, _dominant_hue_name


class TestBallClassification(unittest.TestCase):

    def test_pure_white_cue_ball(self):
        img = np.full((100, 100, 3), 240, dtype=np.uint8)
        label = classify_ball(img, 50, 50, 45)
        self.assertEqual(label, "cue")

    def test_measles_red_dot_cue_ball(self):
        img = np.full((100, 100, 3), 240, dtype=np.uint8)
        dot_centers = [(30, 30), (70, 30), (50, 50), (30, 70), (70, 70), (50, 20)]
        for cx, cy in dot_centers:
            cv2.circle(img, (cx, cy), 5, (0, 0, 220), -1)

        white_ratio = get_ball_white_ratio(img)
        self.assertGreater(white_ratio, 0.80)

        label = classify_ball(img, 50, 50, 45)
        self.assertEqual(label, "cue")

    def test_solid_red_ball_classification(self):
        # Red ball (BGR: 0, 0, 200)
        img = np.full((100, 100, 3), (0, 0, 200), dtype=np.uint8)
        label = classify_ball(img, 50, 50, 45)
        self.assertEqual(label, "solid-red")

    def test_dominant_hue_name_shape_handling(self):
        # 3D array
        hsv_3d = np.zeros((20, 20, 3), dtype=np.uint8)
        hsv_3d[:, :, 0] = 100 # Blue
        hsv_3d[:, :, 1] = 200
        hsv_3d[:, :, 2] = 200
        self.assertEqual(_dominant_hue_name(hsv_3d), "blue")

        # 2D array (filtered pixels)
        hsv_2d = hsv_3d[hsv_3d[:, :, 1] > 100]
        self.assertEqual(_dominant_hue_name(hsv_2d), "blue")


if __name__ == "__main__":
    unittest.main()

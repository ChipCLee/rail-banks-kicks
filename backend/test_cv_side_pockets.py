"""
Unit test suite verifying _detect_long_rails_by_side_pockets on table image fixtures.
"""
import os
import unittest
import cv2
from cv_module import _detect_felt_contour, _order_corners, _detect_long_rails_by_side_pockets, detect_table_and_warp


class TestCVSidePocketOrientation(unittest.TestCase):

    def setUp(self):
        self.base_dir = os.path.dirname(__file__)
        self.root_dir = os.path.dirname(self.base_dir)

    def test_example_1_landscape_detection(self):
        ex1_path = os.path.join(self.root_dir, "example_1.jpg")
        if not os.path.exists(ex1_path):
            self.skipTest("example_1.jpg not found in workspace root")

        img = cv2.imread(ex1_path)
        contour = _detect_felt_contour(img, felt_color="auto")
        self.assertIsNotNone(contour, "_detect_felt_contour returned None on example_1.jpg")

        rect = _order_corners(contour)
        is_portrait = _detect_long_rails_by_side_pockets(img, rect)
        
        # example_1.jpg has long rails horizontal at top/bottom -> is_portrait should be False
        self.assertFalse(is_portrait, "example_1.jpg should be identified as landscape (is_portrait=False)")

        warped, H, dims, is_p = detect_table_and_warp(img)
        self.assertIsNotNone(warped)
        self.assertEqual(warped.shape, (5080, 10160, 3))

    def test_example_portrait_detection_and_horizontal_rotation(self):
        ex_path = os.path.join(self.root_dir, "example.jpg")
        if not os.path.exists(ex_path):
            self.skipTest("example.jpg not found in workspace root")

        img = cv2.imread(ex_path)
        contour = _detect_felt_contour(img, felt_color="auto")
        self.assertIsNotNone(contour, "_detect_felt_contour returned None on example.jpg")

        rect = _order_corners(contour)
        # 1. Raw photo has long rails vertical on Left & Right -> is_portrait should be True
        is_portrait_raw = _detect_long_rails_by_side_pockets(img, rect)
        self.assertTrue(is_portrait_raw, "raw example.jpg should be identified as portrait (is_portrait=True)")

        # 2. After table detection and 90° clockwise rotation, the long rails should be HORIZONTAL
        warped, H, dims, is_p = detect_table_and_warp(img)
        self.assertIsNotNone(warped)
        self.assertEqual(warped.shape, (5080, 10160, 3))

        contour_warped = _detect_felt_contour(warped, felt_color="auto")
        rect_warped = _order_corners(contour_warped)
        is_portrait_warped = _detect_long_rails_by_side_pockets(warped, rect_warped)
        self.assertFalse(is_portrait_warped, "after rotation, example.jpg long rails must be horizontal (is_portrait=False)")


if __name__ == "__main__":
    unittest.main()

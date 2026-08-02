"""
Unit test suite verifying _detect_long_rails_by_side_pockets on table image fixtures
using top-view perspective warping to test midpoints on both horizontal and vertical edges.
"""
import os
import unittest
import cv2
from cv_module import _detect_felt_contour, _order_corners, _detect_long_rails_by_side_pockets, detect_table_and_warp


class TestCVSidePocketOrientation(unittest.TestCase):

    def setUp(self):
        self.base_dir = os.path.dirname(__file__)
        self.root_dir = os.path.dirname(self.base_dir)

    def test_example_jpg_landscape_detection(self):
        ex_path = os.path.join(self.root_dir, "example.jpg")
        if not os.path.exists(ex_path):
            self.skipTest("example.jpg not found in workspace root")

        img = cv2.imread(ex_path)
        contour = _detect_felt_contour(img, felt_color="auto")
        self.assertIsNotNone(contour, "_detect_felt_contour returned None on example.jpg")

        rect = _order_corners(contour)
        is_portrait = _detect_long_rails_by_side_pockets(img, rect)
        
        # example.jpg in perspective top-view has long rails horizontal at top/bottom -> is_portrait should be False
        self.assertFalse(is_portrait, "example.jpg should be identified with is_portrait=False")

        warped, H, dims, is_p = detect_table_and_warp(img)
        self.assertIsNotNone(warped)
        self.assertEqual(warped.shape, (5080, 10160, 3))

    def test_example_1_portrait_detection(self):
        ex1_path = os.path.join(self.root_dir, "example_1.jpg")
        if not os.path.exists(ex1_path):
            self.skipTest("example_1.jpg not found in workspace root")

        img = cv2.imread(ex1_path)
        contour = _detect_felt_contour(img, felt_color="auto")
        self.assertIsNotNone(contour, "_detect_felt_contour returned None on example_1.jpg")

        rect = _order_corners(contour)
        is_portrait = _detect_long_rails_by_side_pockets(img, rect)

        # example_1.jpg in perspective top-view has long rails vertical on left/right -> is_portrait should be True
        self.assertTrue(is_portrait, "example_1.jpg should be identified with is_portrait=True")

        warped, H, dims, is_p = detect_table_and_warp(img)
        self.assertIsNotNone(warped)
        self.assertEqual(warped.shape, (5080, 10160, 3))


if __name__ == "__main__":
    unittest.main()

"""
Unit tests for backend/geometry.py
"""
import unittest
from models import Ball, Pocket, TableDims
from geometry import (
    perp_distance_point_to_segment,
    is_path_clear,
    find_direct_shots,
    find_bank_shots,
)
from cv_module import build_pocket_list

DIMS = TableDims(width=2540.0, height=1270.0)
POCKETS = build_pocket_list(DIMS)


class TestGeometry(unittest.TestCase):

    def test_perp_distance_point_to_segment(self):
        d = perp_distance_point_to_segment(5.0, 5.0, 0.0, 0.0, 10.0, 0.0)
        self.assertAlmostEqual(d, 5.0, places=5)

    def test_is_path_clear(self):
        cue = Ball(id="cue", label="cue", x=500.0, y=635.0)
        obj = Ball(id="obj1", label="solid-red", x=1500.0, y=635.0)
        blocker = Ball(id="obj2", label="eight", x=1000.0, y=635.0)
        
        # Path blocked by blocker
        all_balls = [cue, obj, blocker]
        self.assertFalse(is_path_clear((cue.x, cue.y), (obj.x, obj.y), all_balls, {cue.id, obj.id}))

        # Path clear without blocker
        self.assertTrue(is_path_clear((cue.x, cue.y), (obj.x, obj.y), [cue, obj], {cue.id, obj.id}))

    def test_find_direct_shot(self):
        cue = Ball(id="cue", label="cue", x=1000.0, y=635.0)
        obj = Ball(id="obj1", label="solid-red", x=500.0, y=635.0)
        
        direct_shots = find_direct_shots(cue, [obj], POCKETS, [cue, obj])
        self.assertEqual(len(direct_shots), 1)
        self.assertEqual(direct_shots[0].pocket_id, "ML")
        self.assertEqual(direct_shots[0].shot_type, "direct")
        self.assertEqual(direct_shots[0].ease_score, 0.0)

    def test_find_bank_shot(self):
        # Cue at (800, 980.8), Obj at (1800, 900) -> ray hits RIGHT rail at y=840.2 (valid cushion), reflects to ML (0, 635)
        cue = Ball(id="cue", label="cue", x=800.0, y=980.8)
        obj = Ball(id="obj1", label="eight", x=1800.0, y=900.0)
        
        bank_shots = find_bank_shots(cue, [obj], POCKETS, [cue, obj], DIMS.width, DIMS.height)
        right_banks = [s for s in bank_shots if s.rail == "RIGHT"]
        self.assertGreaterEqual(len(right_banks), 1)
        self.assertEqual(right_banks[0].pocket_id, "ML")


if __name__ == "__main__":
    unittest.main()

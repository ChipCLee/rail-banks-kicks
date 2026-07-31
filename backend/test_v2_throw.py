"""
Unit tests for v2 Feature 4: Cushion Throw Modelling
"""
import unittest
from cushion_throw import calculate_cushion_throw
from models import Ball, TableDims
from cv_module import build_pocket_list
from geometry import find_bank_shots

DIMS = TableDims(width=2540.0, height=1270.0)
POCKETS = build_pocket_list(DIMS)


class TestCushionThrow(unittest.TestCase):

    def test_perpendicular_bank_has_zero_throw(self):
        # At 90° angle of incidence, throw is 0
        throw_corr, adj_rebound = calculate_cushion_throw(90.0)
        self.assertEqual(throw_corr, 0.0)
        self.assertEqual(adj_rebound, 90.0)

    def test_shallow_bank_has_throw_correction(self):
        # At 45° angle, throw should be ~3.5°
        throw_corr, adj_rebound = calculate_cushion_throw(45.0)
        self.assertGreater(throw_corr, 3.0)
        self.assertLess(throw_corr, 5.0)
        self.assertAlmostEqual(adj_rebound, 45.0 - throw_corr, places=2)

    def test_bank_shot_includes_throw_fields(self):
        cue = Ball(id="cue", label="cue", x=800.0, y=980.8)
        obj = Ball(id="obj1", label="eight", x=1800.0, y=900.0)
        
        bank_shots = find_bank_shots(cue, [obj], POCKETS, [cue, obj], DIMS.width, DIMS.height)
        self.assertGreaterEqual(len(bank_shots), 1)
        shot = bank_shots[0]
        self.assertIsNotNone(shot.throw_correction_deg)
        self.assertIsNotNone(shot.adjusted_rebound_angle_deg)


if __name__ == "__main__":
    unittest.main()

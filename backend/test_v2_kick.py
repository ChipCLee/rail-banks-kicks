"""
Unit tests for v2 Feature 3: One-Rail Kick Shot Detection & Diamond System
"""
import unittest
from models import Ball, TableDims
from cv_module import build_pocket_list
from v2_kick_shots import find_kick_shots, _compute_diamond_label

DIMS = TableDims(width=2540.0, height=1270.0)
POCKETS = build_pocket_list(DIMS)


class TestV2KickShots(unittest.TestCase):

    def test_compute_diamond_label(self):
        # 2.5 diamonds from TL corner on TOP rail (2540 x 1270)
        # TOP rail length is 2540, half length is 1270, spacing is 1270 / 4 = 317.5
        # 2.5 diamonds = 2.5 * 317.5 = 793.75 mm
        label = _compute_diamond_label(793.75, 1270.0, "TOP", DIMS.width, DIMS.height)
        self.assertIn("2.5 diamonds", label)
        self.assertIn("TOP", label)

    def test_find_kick_shot_worked_example(self):
        # Scenario from SPEC §3.1: 8-ball is blocked from direct hit.
        # Cue at (400, 900), 8-ball at (1800, 635)
        # Aim cue at TOP rail ~ 2.5 diamond mark (x ≈ 1270 mm)
        cue = Ball(id="cue", label="cue", x=400.0, y=900.0)
        obj = Ball(id="obj_eight", label="eight", x=1800.0, y=635.0)

        # Place a blocker directly between cue and obj to block direct hit
        blocker = Ball(id="obj1", label="solid-red", x=1100.0, y=767.5)

        all_balls = [cue, obj, blocker]
        kick_shots = find_kick_shots(cue, [obj], POCKETS, all_balls, DIMS.width, DIMS.height)

        self.assertGreaterEqual(len(kick_shots), 1)
        top_kicks = [s for s in kick_shots if s.rail == "TOP"]
        self.assertGreaterEqual(len(top_kicks), 1)
        self.assertEqual(top_kicks[0].shot_type, "one_rail_kick")
        self.assertIn("diamonds", top_kicks[0].diamond_label)


if __name__ == "__main__":
    unittest.main()

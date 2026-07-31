"""
v2 Feature 3: One-Rail Kick Shot Detection & Diamond System.

Implements SPEC.md §Feature 3:
  - Cue ball hits a rail cushion first, bounces off, hits the object ball, and object ball pockets directly.
  - Expresses rail contact in standard diamond marker units.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

from models import Ball, Pocket, KickShot, Point, RailId, TableDims
from geometry import (
    is_path_clear,
    _is_contact_point_valid,
    _ray_hits_pocket,
    _vec_norm,
    _vec_len,
    _vec_sub,
    POCKET_CAPTURE_RADIUS_MM,
)
from cv_module import build_pocket_list

RAILS: List[RailId] = ["TOP", "BOTTOM", "LEFT", "RIGHT"]


def _reflect_point_across_rail(
    px: float, py: float, rail: str, table_width: float, table_height: float
) -> Tuple[float, float]:
    """Reflect a 2D point across a given rail wall."""
    if rail == "LEFT":
        return (-px, py)
    elif rail == "RIGHT":
        return (2 * table_width - px, py)
    elif rail == "BOTTOM":
        return (px, -py)
    else:  # TOP
        return (px, 2 * table_height - py)


def _compute_diamond_label(
    px: float, py: float, rail: str, table_width: float, table_height: float
) -> str:
    """
    Convert rail contact coordinates to diamond marker units per SPEC §3.2.
    Diamonds are 0 to 4 along each half-rail.
    """
    if rail in ("TOP", "BOTTOM"):
        rail_len = table_width
        half_len = rail_len / 2.0
        # 4 diamonds per half-rail
        diamond_spacing = half_len / 4.0
        if px <= half_len:
            diamonds = round(px / diamond_spacing, 1)
            corner_ref = "TL" if rail == "TOP" else "BL"
        else:
            diamonds = round((table_width - px) / diamond_spacing, 1)
            corner_ref = "TR" if rail == "TOP" else "BR"
    else:  # LEFT, RIGHT
        rail_len = table_height
        half_len = rail_len / 2.0
        diamond_spacing = half_len / 4.0
        if py <= half_len:
            diamonds = round(py / diamond_spacing, 1)
            corner_ref = "BL" if rail == "LEFT" else "BR"
        else:
            diamonds = round((table_height - py) / diamond_spacing, 1)
            corner_ref = "TL" if rail == "LEFT" else "TR"

    return f"{diamonds} diamonds from {corner_ref} on {rail} rail"


def find_kick_shots(
    cue_ball: Ball,
    object_balls: List[Ball],
    pockets: List[Pocket],
    all_balls: List[Ball],
    table_width: float,
    table_height: float,
) -> List[KickShot]:
    """
    Find all valid one-rail kick shots:
      Cue ball -> Rail contact P -> Object ball O -> Pocket
    """
    results: List[KickShot] = []
    cue_pos = (cue_ball.x, cue_ball.y)

    for obj in object_balls:
        obj_pos = (obj.x, obj.y)
        excluded = {cue_ball.id, obj.id}

        # Step 1 — Check if object ball can travel directly to any pocket
        for pocket in pockets:
            pkt_pos = (pocket.x, pocket.y)
            # Vector obj -> pocket
            raw_op = _vec_sub(pkt_pos, obj_pos)
            op_norm = _vec_norm(raw_op)
            if _vec_len(op_norm) < 1e-9:
                continue

            # Object -> Pocket path must be clear
            if not is_path_clear(obj_pos, pkt_pos, all_balls, excluded):
                continue

            # Step 2 — For each rail, check if cue ball can bounce off rail to hit object ball
            for rail in RAILS:
                # Reflect object ball position across rail
                obj_mirrored = _reflect_point_across_rail(obj.x, obj.y, rail, table_width, table_height)
                
                # Ray from cue ball to mirrored object ball
                ray_dir = _vec_norm(_vec_sub(obj_mirrored, cue_pos))
                if _vec_len(ray_dir) < 1e-9:
                    continue

                dx, dy = ray_dir

                # Intersection of cue -> mirrored obj ray with rail
                if rail == "LEFT":
                    if dx >= 0: continue
                    t = -cue_ball.x / dx
                    px, py = 0.0, cue_ball.y + t * dy
                elif rail == "RIGHT":
                    if dx <= 0: continue
                    t = (table_width - cue_ball.x) / dx
                    px, py = table_width, cue_ball.y + t * dy
                elif rail == "BOTTOM":
                    if dy >= 0: continue
                    t = -cue_ball.y / dy
                    px, py = cue_ball.x + t * dx, 0.0
                else:  # TOP
                    if dy <= 0: continue
                    t = (table_height - cue_ball.y) / dy
                    px, py = cue_ball.x + t * dx, table_height

                contact_pos = (px, py)

                # Validate contact point (must be on cushion, not in pocket opening)
                if not _is_contact_point_valid(px, py, rail, table_width, table_height):
                    continue

                # Check 1: Cue -> Rail contact path clear
                if not is_path_clear(cue_pos, contact_pos, all_balls, excluded):
                    continue

                # Check 2: Rail contact -> Object ball path clear
                if not is_path_clear(contact_pos, obj_pos, all_balls, excluded):
                    continue

                # Compute incident angle at rail
                normal = (1.0, 0.0) if rail in ("LEFT", "RIGHT") else (0.0, 1.0)
                cos_angle = abs(dx * normal[0] + dy * normal[1])
                bank_angle_deg = math.degrees(math.acos(min(1.0, cos_angle)))
                ease_score = abs(bank_angle_deg - 90.0)

                diamond_lbl = _compute_diamond_label(px, py, rail, table_width, table_height)

                results.append(KickShot(
                    cue_ball=Point(x=cue_ball.x, y=cue_ball.y),
                    object_ball_id=obj.id,
                    object_ball_label=obj.label,
                    rail=rail,  # type: ignore[arg-type]
                    contact_point=Point(x=round(px, 1), y=round(py, 1)),
                    diamond_label=diamond_lbl,
                    path=[
                        Point(x=cue_ball.x, y=cue_ball.y),
                        Point(x=round(px, 1), y=round(py, 1)),
                        Point(x=obj.x, y=obj.y),
                        Point(x=pocket.x, y=pocket.y),
                    ],
                    bank_angle_deg=round(bank_angle_deg, 2),
                    ease_score=round(ease_score, 2),
                    pocket_id=pocket.id,  # type: ignore[arg-type]
                ))

    # Sort kick shots by ease score
    results.sort(key=lambda s: s.ease_score)
    return results

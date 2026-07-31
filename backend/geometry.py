"""
Geometry module for Rail-Kick.

Implements:
  - Perpendicular distance from a point to a line segment
  - Path obstruction check (ghost-ball clearance)
  - Direct shot detection
  - One-bank shot detection (object ball reflection off rail)
  - Integrated cushion throw correction (v2 Feature 4)

All coordinates are in mm in table space:
  x: 0 (left) → table_width  (right)
  y: 0 (bottom) → table_height (top)

SPEC.md §2.2 Algorithm & §Feature 4 reference.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

from models import Ball, Pocket, DirectShot, BankShot, Point, PocketId, RailId
from cushion_throw import calculate_cushion_throw

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BALL_DIAMETER_MM = 57.15
BALL_RADIUS_MM = BALL_DIAMETER_MM / 2.0

CORNER_POCKET_RADIUS_MM = 57.0
SIDE_POCKET_RADIUS_MM = 63.0
POCKET_CAPTURE_RADIUS_MM = SIDE_POCKET_RADIUS_MM


def _vec_sub(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return (a[0] - b[0], a[1] - b[1])


def _vec_dot(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _vec_len(v: Tuple[float, float]) -> float:
    return math.hypot(v[0], v[1])


def _vec_norm(v: Tuple[float, float]) -> Tuple[float, float]:
    length = _vec_len(v)
    if length < 1e-9:
        return (0.0, 0.0)
    return (v[0] / length, v[1] / length)


def _vec_scale(v: Tuple[float, float], s: float) -> Tuple[float, float]:
    return (v[0] * s, v[1] * s)


def _vec_add(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return (a[0] + b[0], a[1] + b[1])


def perp_distance_point_to_segment(
    px: float, py: float,
    ax: float, ay: float,
    bx: float, by: float,
) -> float:
    dx = bx - ax
    dy = by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-12:
        return math.hypot(px - ax, py - ay)

    t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))

    cx = ax + t * dx
    cy = ay + t * dy
    return math.hypot(px - cx, py - cy)


def is_path_clear(
    start: Tuple[float, float],
    end: Tuple[float, float],
    all_balls: List[Ball],
    excluded_ids: set[str],
) -> bool:
    ax, ay = start
    bx, by = end
    for ball in all_balls:
        if ball.id in excluded_ids:
            continue
        dist = perp_distance_point_to_segment(ball.x, ball.y, ax, ay, bx, by)
        if dist < BALL_DIAMETER_MM:
            return False
    return True


def _is_contact_point_valid(
    px: float, py: float,
    rail: str,
    table_width: float,
    table_height: float,
) -> bool:
    cp_r = CORNER_POCKET_RADIUS_MM
    sp_r = SIDE_POCKET_RADIUS_MM

    if rail in ("LEFT", "RIGHT"):
        if py < cp_r or py > table_height - cp_r:
            return False
        mid = table_height / 2.0
        if mid - sp_r <= py <= mid + sp_r:
            return False
    else:
        if px < cp_r or px > table_width - cp_r:
            return False
        mid = table_width / 2.0
        if mid - sp_r <= px <= mid + sp_r:
            return False
    return True


def _ray_hits_pocket(
    ox: float, oy: float,
    dx: float, dy: float,
    pocket: Pocket,
) -> bool:
    tx = pocket.x - ox
    ty = pocket.y - oy
    t = tx * dx + ty * dy
    if t < 0:
        return False
    cx = ox + t * dx
    cy = oy + t * dy
    dist = math.hypot(cx - pocket.x, cy - pocket.y)
    return dist <= POCKET_CAPTURE_RADIUS_MM


def _ray_pocket_intersection(
    ox: float, oy: float,
    dx: float, dy: float,
    pockets: List[Pocket],
) -> Optional[Pocket]:
    for pocket in pockets:
        if _ray_hits_pocket(ox, oy, dx, dy, pocket):
            return pocket
    return None


def _reflect_direction(
    dx: float, dy: float, rail: str
) -> Tuple[float, float]:
    if rail in ("LEFT", "RIGHT"):
        return (-dx, dy)
    else:
        return (dx, -dy)


def _ray_rail_intersection(
    ox: float, oy: float,
    dx: float, dy: float,
    rail: str,
    table_width: float,
    table_height: float,
) -> Optional[Tuple[float, float]]:
    EPS = 1e-9
    if rail == "LEFT":
        if dx >= -EPS:
            return None
        t = -ox / dx
    elif rail == "RIGHT":
        if dx <= EPS:
            return None
        t = (table_width - ox) / dx
    elif rail == "BOTTOM":
        if dy >= -EPS:
            return None
        t = -oy / dy
    else:
        if dy <= EPS:
            return None
        t = (table_height - oy) / dy

    if t <= EPS:
        return None
    return (ox + t * dx, oy + t * dy)


def find_direct_shots(
    cue_ball: Ball,
    object_balls: List[Ball],
    pockets: List[Pocket],
    all_balls: List[Ball],
) -> List[DirectShot]:
    results: List[DirectShot] = []
    cue_pos = (cue_ball.x, cue_ball.y)

    for obj in object_balls:
        obj_pos = (obj.x, obj.y)
        excluded = {cue_ball.id, obj.id}
        if not is_path_clear(cue_pos, obj_pos, all_balls, excluded):
            continue

        raw_dx = obj.x - cue_ball.x
        raw_dy = obj.y - cue_ball.y
        d = _vec_norm((raw_dx, raw_dy))
        if _vec_len(d) < 1e-9:
            continue

        for pocket in pockets:
            hit = _ray_hits_pocket(obj.x, obj.y, d[0], d[1], pocket)
            if not hit:
                continue

            pkt_pos = (pocket.x, pocket.y)
            if not is_path_clear(obj_pos, pkt_pos, all_balls, excluded):
                continue

            results.append(DirectShot(
                cue_ball=Point(x=cue_ball.x, y=cue_ball.y),
                object_ball_id=obj.id,
                object_ball_label=obj.label,
                path=[
                    Point(x=cue_ball.x, y=cue_ball.y),
                    Point(x=obj.x, y=obj.y),
                    Point(x=pocket.x, y=pocket.y),
                ],
                ease_score=0.0,
                pocket_id=pocket.id,  # type: ignore[arg-type]
            ))

    results.sort(key=lambda s: s.pocket_id)
    return results


RAILS: List[RailId] = ["TOP", "BOTTOM", "LEFT", "RIGHT"]


def find_bank_shots(
    cue_ball: Ball,
    object_balls: List[Ball],
    pockets: List[Pocket],
    all_balls: List[Ball],
    table_width: float,
    table_height: float,
) -> List[BankShot]:
    results: List[BankShot] = []
    cue_pos = (cue_ball.x, cue_ball.y)

    for obj in object_balls:
        obj_pos = (obj.x, obj.y)
        excluded = {cue_ball.id, obj.id}
        if not is_path_clear(cue_pos, obj_pos, all_balls, excluded):
            continue

        raw = _vec_sub(obj_pos, cue_pos)
        d = _vec_norm(raw)
        if _vec_len(d) < 1e-9:
            continue

        dx, dy = d

        for rail in RAILS:
            contact = _ray_rail_intersection(
                obj.x, obj.y, dx, dy, rail, table_width, table_height
            )
            if contact is None:
                continue
            px, py = contact

            if not _is_contact_point_valid(px, py, rail, table_width, table_height):
                continue

            rdx, rdy = _reflect_direction(dx, dy, rail)

            normal = (1.0, 0.0) if rail in ("LEFT", "RIGHT") else (0.0, 1.0)
            cos_angle = abs(dx * normal[0] + dy * normal[1])
            bank_angle_deg = math.degrees(math.acos(min(1.0, cos_angle)))
            ease_score = abs(bank_angle_deg - 90.0)

            # v2 Feature 4: Calculate cushion throw
            throw_corr, adj_rebound = calculate_cushion_throw(bank_angle_deg)

            target_pocket = _ray_pocket_intersection(px, py, rdx, rdy, pockets)
            if target_pocket is None:
                continue

            pkt_pos = (target_pocket.x, target_pocket.y)
            if not is_path_clear((px, py), pkt_pos, all_balls, excluded):
                continue

            results.append(BankShot(
                cue_ball=Point(x=cue_ball.x, y=cue_ball.y),
                object_ball_id=obj.id,
                object_ball_label=obj.label,
                rail=rail,  # type: ignore[arg-type]
                contact_point=Point(x=round(px, 1), y=round(py, 1)),
                path=[
                    Point(x=cue_ball.x, y=cue_ball.y),
                    Point(x=obj.x, y=obj.y),
                    Point(x=round(px, 1), y=round(py, 1)),
                    Point(x=target_pocket.x, y=target_pocket.y),
                ],
                bank_angle_deg=round(bank_angle_deg, 2),
                throw_correction_deg=throw_corr,
                adjusted_rebound_angle_deg=adj_rebound,
                ease_score=round(ease_score, 2),
                pocket_id=target_pocket.id,  # type: ignore[arg-type]
            ))

    results.sort(key=lambda s: s.ease_score)
    return results

"""
Geometry module for Rail-Kick.

Implements:
  - Perpendicular distance from a point to a line segment
  - Path obstruction check (ghost-ball clearance)
  - Direct shot detection
  - One-bank shot detection (object ball reflection off rail)

All coordinates are in mm in table space:
  x: 0 (left) → table_width  (right)
  y: 0 (bottom) → table_height (top)

SPEC.md §2.2 Algorithm reference.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

from models import Ball, Pocket, DirectShot, BankShot, Point, PocketId, RailId

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BALL_DIAMETER_MM = 57.15
BALL_RADIUS_MM = BALL_DIAMETER_MM / 2.0

# Pocket opening radii (half the physical opening width)
CORNER_POCKET_RADIUS_MM = 57.0
SIDE_POCKET_RADIUS_MM = 63.0

# Pocket check tolerance — how close the reflected ray must come to pocket centre
POCKET_CAPTURE_RADIUS_MM = SIDE_POCKET_RADIUS_MM


# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Perpendicular distance from point P to the line segment AB
# ---------------------------------------------------------------------------

def perp_distance_point_to_segment(
    px: float, py: float,
    ax: float, ay: float,
    bx: float, by: float,
) -> float:
    """
    Returns the minimum distance from point (px, py) to the finite
    line segment from (ax, ay) to (bx, by).
    """
    dx = bx - ax
    dy = by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-12:
        # Degenerate segment — treat as point distance
        return math.hypot(px - ax, py - ay)

    # Project P onto the line AB, clamped to [0, 1]
    t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))

    # Closest point on segment
    cx = ax + t * dx
    cy = ay + t * dy
    return math.hypot(px - cx, py - cy)


# ---------------------------------------------------------------------------
# Path clear check
# ---------------------------------------------------------------------------

def is_path_clear(
    start: Tuple[float, float],
    end: Tuple[float, float],
    all_balls: List[Ball],
    excluded_ids: set[str],
) -> bool:
    """
    Returns True if the line segment (start → end) is not obstructed by
    any ball whose ID is not in excluded_ids.

    Obstruction: perpendicular distance from ball centre to segment < BALL_DIAMETER_MM.
    (ghost-ball clearance check per SPEC §2.2 Step 1)
    """
    ax, ay = start
    bx, by = end
    for ball in all_balls:
        if ball.id in excluded_ids:
            continue
        dist = perp_distance_point_to_segment(ball.x, ball.y, ax, ay, bx, by)
        if dist < BALL_DIAMETER_MM:
            return False
    return True


# ---------------------------------------------------------------------------
# Rail contact point validity
# ---------------------------------------------------------------------------

def _is_contact_point_valid(
    px: float, py: float,
    rail: str,
    table_width: float,
    table_height: float,
) -> bool:
    """
    Check that contact point P lies in the active cushion range:
    not inside a corner or side pocket opening.
    SPEC §2.2 Step 4.
    """
    cp_r = CORNER_POCKET_RADIUS_MM
    sp_r = SIDE_POCKET_RADIUS_MM

    if rail in ("LEFT", "RIGHT"):
        # Valid y range
        if py < cp_r or py > table_height - cp_r:
            return False
        # Exclude side pocket zone
        mid = table_height / 2.0
        if mid - sp_r <= py <= mid + sp_r:
            return False
    else:  # TOP, BOTTOM
        if px < cp_r or px > table_width - cp_r:
            return False
        mid = table_width / 2.0
        if mid - sp_r <= px <= mid + sp_r:
            return False
    return True


# ---------------------------------------------------------------------------
# Ray → pocket check
# ---------------------------------------------------------------------------

def _ray_hits_pocket(
    ox: float, oy: float,
    dx: float, dy: float,
    pocket: Pocket,
) -> bool:
    """
    Check if the ray starting at (ox, oy) in direction (dx, dy) passes
    within POCKET_CAPTURE_RADIUS_MM of the pocket centre.
    Only counts if the pocket is ahead of the ray (t > 0).
    """
    # Vector from ray origin to pocket centre
    tx = pocket.x - ox
    ty = pocket.y - oy
    # Project onto ray direction
    t = tx * dx + ty * dy
    if t < 0:
        return False
    # Closest point on ray to pocket centre
    cx = ox + t * dx
    cy = oy + t * dy
    dist = math.hypot(cx - pocket.x, cy - pocket.y)
    return dist <= POCKET_CAPTURE_RADIUS_MM


def _ray_pocket_intersection(
    ox: float, oy: float,
    dx: float, dy: float,
    pockets: List[Pocket],
) -> Optional[Pocket]:
    """
    Return the first pocket the ray hits, or None.
    """
    for pocket in pockets:
        if _ray_hits_pocket(ox, oy, dx, dy, pocket):
            return pocket
    return None


# ---------------------------------------------------------------------------
# Rail reflection
# ---------------------------------------------------------------------------

def _reflect_direction(
    dx: float, dy: float, rail: str
) -> Tuple[float, float]:
    """Reflect direction vector off a rail."""
    if rail in ("LEFT", "RIGHT"):
        return (-dx, dy)
    else:  # TOP, BOTTOM
        return (dx, -dy)


def _ray_rail_intersection(
    ox: float, oy: float,
    dx: float, dy: float,
    rail: str,
    table_width: float,
    table_height: float,
) -> Optional[Tuple[float, float]]:
    """
    Find where the ray (ox, oy, dx, dy) intersects the given rail wall.
    Returns None if the ray moves away from that rail.
    """
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
    else:  # TOP
        if dy <= EPS:
            return None
        t = (table_height - oy) / dy

    if t <= EPS:
        return None
    return (ox + t * dx, oy + t * dy)


# ---------------------------------------------------------------------------
# Direct shot detection (SPEC §Feature 2 / v1)
# ---------------------------------------------------------------------------

def find_direct_shots(
    cue_ball: Ball,
    object_balls: List[Ball],
    pockets: List[Pocket],
    all_balls: List[Ball],
) -> List[DirectShot]:
    """
    For each object ball, check if the cue ball can reach it directly (no rail)
    AND if the object ball can then travel directly into any pocket.
    """
    results: List[DirectShot] = []
    cue_pos = (cue_ball.x, cue_ball.y)

    for obj in object_balls:
        obj_pos = (obj.x, obj.y)

        # Step 1 — Can cue ball reach object ball?
        excluded = {cue_ball.id, obj.id}
        if not is_path_clear(cue_pos, obj_pos, all_balls, excluded):
            continue

        # Direction cue→object (object ball departure direction after being struck)
        raw_dx = obj.x - cue_ball.x
        raw_dy = obj.y - cue_ball.y
        d = _vec_norm((raw_dx, raw_dy))
        if _vec_len(d) < 1e-9:
            continue

        # Check each pocket: does the object ball travel directly to it?
        for pocket in pockets:
            # Direction object→pocket
            pd = _vec_norm((pocket.x - obj.x, pocket.y - obj.y))
            # Alignment: cue→obj direction must point toward pocket
            # Use ray from object in departure direction
            hit = _ray_hits_pocket(obj.x, obj.y, d[0], d[1], pocket)
            if not hit:
                continue

            # Obstruction check: object ball → pocket
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

    # Direct shots are easiest (ease_score=0), sort stable by pocket id for determinism
    results.sort(key=lambda s: s.pocket_id)
    return results


# ---------------------------------------------------------------------------
# Bank shot detection (SPEC §Feature 2 / v1)
# ---------------------------------------------------------------------------

RAILS: List[RailId] = ["TOP", "BOTTOM", "LEFT", "RIGHT"]


def find_bank_shots(
    cue_ball: Ball,
    object_balls: List[Ball],
    pockets: List[Pocket],
    all_balls: List[Ball],
    table_width: float,
    table_height: float,
) -> List[BankShot]:
    """
    For each (cue_ball, object_ball, rail) triple:
      1. Check cue ball can reach object ball directly.
      2. Compute object ball departure direction.
      3. Find rail contact point using reflection.
      4. Validate contact point (not in pocket zone).
      5. Check reflected ray reaches a pocket.
      6. Verify post-rail path is clear.

    SPEC §2.2 Algorithm.
    """
    results: List[BankShot] = []
    cue_pos = (cue_ball.x, cue_ball.y)

    for obj in object_balls:
        obj_pos = (obj.x, obj.y)

        # Step 1 — cue ball → object ball must be clear
        excluded = {cue_ball.id, obj.id}
        if not is_path_clear(cue_pos, obj_pos, all_balls, excluded):
            continue

        # Step 2 — departure direction (ghost-ball: cue→obj direction)
        raw = _vec_sub(obj_pos, cue_pos)
        d = _vec_norm(raw)
        if _vec_len(d) < 1e-9:
            continue

        dx, dy = d

        for rail in RAILS:
            # Step 3 — find where object ball path hits this rail
            contact = _ray_rail_intersection(
                obj.x, obj.y, dx, dy, rail, table_width, table_height
            )
            if contact is None:
                continue
            px, py = contact

            # Step 4 — contact point must be on valid cushion
            if not _is_contact_point_valid(px, py, rail, table_width, table_height):
                continue

            # Reflected direction
            rdx, rdy = _reflect_direction(dx, dy, rail)

            # Compute bank angle (angle of incidence at the rail)
            # Angle between incoming direction and the rail normal
            if rail in ("LEFT", "RIGHT"):
                normal = (1.0, 0.0)
            else:
                normal = (0.0, 1.0)
            cos_angle = abs(dx * normal[0] + dy * normal[1])
            bank_angle_deg = math.degrees(math.acos(min(1.0, cos_angle)))
            ease_score = abs(bank_angle_deg - 90.0)

            # Step 5 — does reflected ray reach a pocket?
            target_pocket = _ray_pocket_intersection(px, py, rdx, rdy, pockets)
            if target_pocket is None:
                continue

            # Step 6 — post-rail path must be clear
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
                ease_score=round(ease_score, 2),
                pocket_id=target_pocket.id,  # type: ignore[arg-type]
            ))

    # Sort by ease_score ascending (closest to 90° first)
    results.sort(key=lambda s: s.ease_score)
    return results

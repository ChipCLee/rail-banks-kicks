"""
Image annotation module for Rail-Kick.

Draws visual overlays on top-down warped table image according to SPEC.md §2.4 & §3.5:
  - Rail diamonds: Small diamond markers + number labels on cushions
  - Cue ball: White circle outline + label "CUE"
  - Object balls: Color-matched outline + label
  - Cue → object path: Blue solid arrow
  - Object → rail contact: Orange dashed arrow
  - Rail contact → pocket: Green dashed arrow
  - Rail contact point: White filled dot
  - Kick shot: Blue solid arrow (cue→rail), Orange solid arrow (rail→obj), Green dashed (obj→pocket), Blue diamond marker
  - Target pocket: Purple ring
"""
from __future__ import annotations

import base64
from typing import List, Optional, Tuple

import cv2
import numpy as np

from models import Ball, Pocket, DiamondMarker, DirectShot, BankShot, KickShot, TableDims


# Color palette in BGR
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_BLUE_ARROW = (235, 130, 0)      # Cue → Object / Cue → Rail
COLOR_ORANGE_ARROW = (0, 140, 255)    # Object → Rail / Rail → Object
COLOR_GREEN_ARROW = (50, 205, 50)     # Rail → Pocket / Object → Pocket
COLOR_YELLOW_TARGET = (0, 215, 255)   # Target ball highlight
COLOR_PURPLE_POCKET = (211, 0, 148)   # Target pocket highlight
COLOR_DIAMOND_BLUE = (255, 191, 0)    # Diamond marker icon
COLOR_RAIL_DIAMOND = (0, 200, 255)    # Rail diamond marker color (Yellow/Cyan)


def _mm_to_px(x_mm: float, y_mm: float, dims: TableDims, img_shape: Tuple[int, int, int]) -> Tuple[int, int]:
    """Convert table mm coordinates (origin bottom-left) to image pixel coordinates (origin top-left)."""
    h_px, w_px = img_shape[0], img_shape[1]
    px_x = int((x_mm / dims.width) * w_px)
    px_y = int(h_px - (y_mm / dims.height) * h_px)
    return (px_x, px_y)


def _draw_dashed_line(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 2,
    dash_len: int = 10,
):
    """Draw a dashed line segment between pt1 and pt2."""
    dist = float(np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]))
    if dist < 1e-3:
        return
    dvec = ((pt2[0] - pt1[0]) / dist, (pt2[1] - pt1[1]) / dist)
    
    curr = 0.0
    drawing = True
    while curr < dist:
        nxt = min(curr + dash_len, dist)
        if drawing:
            p_start = (int(pt1[0] + dvec[0] * curr), int(pt1[1] + dvec[1] * curr))
            p_end = (int(pt1[0] + dvec[0] * nxt), int(pt1[1] + dvec[1] * nxt))
            cv2.line(img, p_start, p_end, color, thickness, cv2.LINE_AA)
        curr = nxt
        drawing = not drawing


def _draw_diamond_marker(img: np.ndarray, pt: Tuple[int, int], size: int = 8, color=COLOR_DIAMOND_BLUE):
    """Draw a diamond shape icon."""
    pts = np.array([
        [pt[0], pt[1] - size],
        [pt[0] + size, pt[1]],
        [pt[0], pt[1] + size],
        [pt[0] - size, pt[1]],
    ], np.int32)
    cv2.fillPoly(img, [pts], color, cv2.LINE_AA)
    cv2.polylines(img, [pts], True, COLOR_WHITE, 1, cv2.LINE_AA)


def annotate_table(
    warped: np.ndarray,
    dims: TableDims,
    pockets: List[Pocket],
    balls: List[Ball],
    direct_shots: List[DirectShot],
    bank_shots: List[BankShot],
    kick_shots: Optional[List[KickShot]] = None,
    diamonds: Optional[List[DiamondMarker]] = None,
    selected_shot_index: Optional[int] = 0,
) -> str:
    """
    Annotate warped image and return base64 JPEG string.
    """
    if kick_shots is None:
        kick_shots = []
    if diamonds is None:
        diamonds = []

    img = warped.copy()

    # Draw rail diamond markers
    for d in diamonds:
        dpx = _mm_to_px(d.x, d.y, dims, img.shape)
        _draw_diamond_marker(img, dpx, size=5, color=COLOR_RAIL_DIAMOND)
        # Small text label
        offset_y = 12 if d.rail == "TOP" else -8 if d.rail == "BOTTOM" else 4
        offset_x = 8 if d.rail == "LEFT" else -14 if d.rail == "RIGHT" else -6
        cv2.putText(
            img,
            str(d.number),
            (dpx[0] + offset_x, dpx[1] + offset_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            COLOR_WHITE,
            1,
            cv2.LINE_AA,
        )

    # Draw pockets
    for pkt in pockets:
        ppx = _mm_to_px(pkt.x, pkt.y, dims, img.shape)
        cv2.circle(img, ppx, 12, (50, 50, 50), -1, cv2.LINE_AA)
        cv2.putText(img, pkt.id, (ppx[0] - 8, ppx[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_WHITE, 1, cv2.LINE_AA)

    # Draw balls
    for ball in balls:
        bpx = _mm_to_px(ball.x, ball.y, dims, img.shape)
        r_px = int((ball.radius_mm / dims.width) * img.shape[1])

        if ball.label == "cue":
            cv2.circle(img, bpx, r_px, COLOR_WHITE, 2, cv2.LINE_AA)
            cv2.putText(img, "CUE", (bpx[0] - 14, bpx[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_WHITE, 1, cv2.LINE_AA)
        else:
            color = (0, 0, 255) if "red" in ball.label else (255, 100, 0) if "blue" in ball.label else (200, 200, 200)
            if ball.label == "eight":
                color = (30, 30, 30)
            cv2.circle(img, bpx, r_px, color, 2, cv2.LINE_AA)
            cv2.putText(img, ball.label, (bpx[0] - 18, bpx[1] + r_px + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_WHITE, 1, cv2.LINE_AA)

    # Combine all shots
    all_shots = list(direct_shots) + list(bank_shots) + list(kick_shots)
    if all_shots and selected_shot_index is not None and 0 <= selected_shot_index < len(all_shots):
        shot = all_shots[selected_shot_index]
        
        # Highlight target pocket
        pocket_obj = next((p for p in pockets if p.id == shot.pocket_id), None)
        if pocket_obj:
            ppx = _mm_to_px(pocket_obj.x, pocket_obj.y, dims, img.shape)
            cv2.circle(img, ppx, 20, COLOR_PURPLE_POCKET, 3, cv2.LINE_AA)

        # Highlight target ball
        target_ball = next((b for b in balls if b.id == shot.object_ball_id), None)
        if target_ball:
            tbpx = _mm_to_px(target_ball.x, target_ball.y, dims, img.shape)
            cv2.circle(img, tbpx, int((target_ball.radius_mm / dims.width) * img.shape[1]), COLOR_YELLOW_TARGET, -1, cv2.LINE_AA)

        # Draw path
        pts_px = [_mm_to_px(pt.x, pt.y, dims, img.shape) for pt in shot.path]
        if shot.shot_type == "direct" and len(pts_px) >= 3:
            # Cue -> Obj
            cv2.arrowedLine(img, pts_px[0], pts_px[1], COLOR_BLUE_ARROW, 2, cv2.LINE_AA)
            # Obj -> Pocket
            _draw_dashed_line(img, pts_px[1], pts_px[2], COLOR_GREEN_ARROW, 2)
        elif shot.shot_type == "one_bank" and len(pts_px) >= 4:
            # Cue -> Obj
            cv2.arrowedLine(img, pts_px[0], pts_px[1], COLOR_BLUE_ARROW, 2, cv2.LINE_AA)
            # Obj -> Rail
            _draw_dashed_line(img, pts_px[1], pts_px[2], COLOR_ORANGE_ARROW, 2)
            # Rail contact point
            cv2.circle(img, pts_px[2], 5, COLOR_WHITE, -1, cv2.LINE_AA)
            # Rail -> Pocket
            _draw_dashed_line(img, pts_px[2], pts_px[3], COLOR_GREEN_ARROW, 2)
        elif shot.shot_type == "one_rail_kick" and len(pts_px) >= 4:
            # Cue -> Rail
            cv2.arrowedLine(img, pts_px[0], pts_px[1], COLOR_BLUE_ARROW, 2, cv2.LINE_AA)
            # Rail -> Obj
            cv2.arrowedLine(img, pts_px[1], pts_px[2], COLOR_ORANGE_ARROW, 2, cv2.LINE_AA)
            # Diamond marker
            _draw_diamond_marker(img, pts_px[1], size=8)
            # Obj -> Pocket
            _draw_dashed_line(img, pts_px[2], pts_px[3], COLOR_GREEN_ARROW, 2)

    # Encode JPEG
    _, buffer = cv2.imencode(".jpg", img)
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return b64_str

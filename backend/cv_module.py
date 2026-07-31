"""
Computer Vision module for Rail-Kick.

Implements:
  1. Table boundary detection (green felt HSV masking → homography)
  2. Ball detection (Gaussian blur + HoughCircles)
  3. Ball classification (HSV color profiling with support for red-dot / measles cue balls)
  4. Diamond marker generation (for rails)
  5. Perspective-corrected top-down image output + Teach Mode fallback

All measurements are in mm in table space.
SPEC.md §Feature 1 – Ball Position Analysis.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

from models import Ball, Pocket, DiamondMarker, TableDims

# ---------------------------------------------------------------------------
# Table dimensions (SPEC §1.2 Step 1)
# ---------------------------------------------------------------------------

TABLE_CONFIGS = {
    "9ft": {"width_mm": 2540.0, "height_mm": 1270.0},
    "8ft": {"width_mm": 2240.0, "height_mm": 1120.0},
    "7ft": {"width_mm": 1981.0, "height_mm":  991.0},
}
DEFAULT_TABLE = "9ft"

BALL_DIAMETER_MM = 57.15
BALL_RADIUS_MM = BALL_DIAMETER_MM / 2.0
CORNER_POCKET_RADIUS_MM = 57.0
SIDE_POCKET_RADIUS_MM = 63.0


# ---------------------------------------------------------------------------
# Step 1 – Table Boundary Detection
# ---------------------------------------------------------------------------

def _detect_felt_contour(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect the largest green-felt region using HSV color masking.
    Returns the 4-point approximated contour or None.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    lower1 = np.array([35, 40, 30])
    upper1 = np.array([95, 255, 220])
    mask = cv2.inRange(hsv, lower1, upper1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

    if len(approx) != 4:
        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect)
        approx = box.reshape(-1, 1, 2).astype(np.float32)

    return approx


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 corner points as [TL, TR, BR, BL] (clockwise from top-left)."""
    pts = pts.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # TL
    rect[2] = pts[np.argmax(s)]   # BR

    diff = np.diff(pts, axis=1).ravel()
    rect[1] = pts[np.argmin(diff)]  # TR
    rect[3] = pts[np.argmax(diff)]  # BL

    return rect


def detect_table_and_warp(
    image_bgr: np.ndarray,
    table_size: str = DEFAULT_TABLE,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[TableDims]]:
    """
    Detect table boundary and return:
      - warped: perspective-corrected top-down BGR image
      - H: homography matrix
      - dims: TableDims(width_mm, height_mm)
    """
    config = TABLE_CONFIGS.get(table_size, TABLE_CONFIGS[DEFAULT_TABLE])
    w_mm = config["width_mm"]
    h_mm = config["height_mm"]

    px_per_mm = 4.0
    w_px = int(w_mm * px_per_mm)
    h_px = int(h_mm * px_per_mm)

    contour = _detect_felt_contour(image_bgr)
    if contour is None:
        return None, None, None

    src_pts = _order_corners(contour)
    dst_pts = np.array([
        [0,    h_px],  # TL
        [w_px, h_px],  # TR
        [w_px, 0   ],  # BR
        [0,    0   ],  # BL
    ], dtype=np.float32)

    H, _ = cv2.findHomography(src_pts, dst_pts)
    if H is None:
        return None, None, None

    warped = cv2.warpPerspective(image_bgr, H, (w_px, h_px))
    dims = TableDims(width=w_mm, height=h_mm)
    return warped, H, dims


def build_pocket_list(dims: TableDims) -> List[Pocket]:
    """Build six pocket positions in table mm space."""
    w = dims.width
    h = dims.height
    return [
        Pocket(id="TL", x=0.0,   y=h,    radius_mm=CORNER_POCKET_RADIUS_MM),
        Pocket(id="TR", x=w,     y=h,    radius_mm=CORNER_POCKET_RADIUS_MM),
        Pocket(id="ML", x=0.0,   y=h/2,  radius_mm=SIDE_POCKET_RADIUS_MM),
        Pocket(id="MR", x=w,     y=h/2,  radius_mm=SIDE_POCKET_RADIUS_MM),
        Pocket(id="BL", x=0.0,   y=0.0,  radius_mm=CORNER_POCKET_RADIUS_MM),
        Pocket(id="BR", x=w,     y=0.0,  radius_mm=CORNER_POCKET_RADIUS_MM),
    ]


def build_diamond_list(dims: TableDims) -> List[DiamondMarker]:
    """
    Build the list of diamond markers along the 4 rails.
    - Long rails (TOP, BOTTOM): 7 diamonds (numbers 1, 2, 3, 4(side), 3, 2, 1)
    - Short rails (LEFT, RIGHT): 3 diamonds (numbers 1, 2, 3)
    """
    w = dims.width
    h = dims.height
    diamonds: List[DiamondMarker] = []

    # TOP & BOTTOM rails (8 segments -> 7 diamonds)
    seg_w = w / 8.0
    for i in range(1, 8):
        x_pos = round(i * seg_w, 1)
        num = i * 0.5 if i <= 4 else (8 - i) * 0.5
        diamonds.append(DiamondMarker(rail="TOP", number=num, x=x_pos, y=h, label=f"{num} TOP"))
        diamonds.append(DiamondMarker(rail="BOTTOM", number=num, x=x_pos, y=0.0, label=f"{num} BTM"))

    # LEFT & RIGHT rails (4 segments -> 3 diamonds)
    seg_h = h / 4.0
    for i in range(1, 4):
        y_pos = round(i * seg_h, 1)
        diamonds.append(DiamondMarker(rail="LEFT", number=float(i), x=0.0, y=y_pos, label=f"{i} LEFT"))
        diamonds.append(DiamondMarker(rail="RIGHT", number=float(i), x=w, y=y_pos, label=f"{i} RIGHT"))

    return diamonds


# ---------------------------------------------------------------------------
# Step 2 – Ball Detection
# ---------------------------------------------------------------------------

def _px_per_mm_from_warped(warped: np.ndarray, dims: TableDims) -> float:
    return warped.shape[1] / dims.width


def detect_balls_on_warped(
    warped: np.ndarray,
    dims: TableDims,
) -> List[Tuple[float, float, float]]:
    px_mm = _px_per_mm_from_warped(warped, dims)
    expected_r_px = BALL_RADIUS_MM * px_mm

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)

    min_r = int(expected_r_px * 0.6)
    max_r = int(expected_r_px * 1.5)
    min_dist = int(expected_r_px * 1.8)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dist,
        param1=60,
        param2=25,
        minRadius=min_r,
        maxRadius=max_r,
    )

    results: List[Tuple[float, float, float]] = []
    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        h_px = warped.shape[0]
        for (cx_px, cy_px, r_px) in circles:
            x_mm = cx_px / px_mm
            y_mm = (h_px - cy_px) / px_mm
            r_mm = r_px / px_mm
            results.append((x_mm, y_mm, r_mm))
    return results


# ---------------------------------------------------------------------------
# Step 3 – Ball Classification
# ---------------------------------------------------------------------------

_HUE_BUCKETS = [
    (0,   10,  "red"),
    (10,  25,  "orange"),
    (25,  35,  "yellow"),
    (35,  85,  "green"),
    (85,  130, "blue"),
    (130, 160, "purple"),
    (160, 180, "red"),
]


def _dominant_hue_name(hsv_roi: np.ndarray) -> str:
    if hsv_roi.size == 0:
        return "unknown"
    hue_channel = hsv_roi[:, :, 0].ravel()
    if len(hue_channel) == 0:
        return "unknown"
    hist = np.bincount(hue_channel.astype(int), minlength=181)
    dominant_hue = int(np.argmax(hist))
    for lo, hi, name in _HUE_BUCKETS:
        if lo <= dominant_hue < hi:
            return name
    return "red"


def get_ball_white_ratio(roi: np.ndarray) -> float:
    if roi.size == 0:
        return 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(float)
    s = hsv[:, :, 1].astype(float)
    white_pixels = (v > 140) & (s < 75)
    return float(np.sum(white_pixels)) / white_pixels.size


def classify_ball(
    warped: np.ndarray,
    cx_px: int,
    cy_px: int,
    r_px: int,
) -> str:
    x1 = max(0, cx_px - r_px)
    y1 = max(0, cy_px - r_px)
    x2 = min(warped.shape[1], cx_px + r_px)
    y2 = min(warped.shape[0], cy_px + r_px)
    roi = warped[y1:y2, x1:x2]

    if roi.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(float)
    s = hsv[:, :, 1].astype(float)
    h = hsv[:, :, 0].astype(float)
    mean_v = float(np.mean(v))

    white_ratio = get_ball_white_ratio(roi)
    red_dots_mask = ((h <= 10) | (h >= 160)) & (s > 80) & (v > 100)
    red_dots_ratio = float(np.sum(red_dots_mask)) / red_dots_mask.size

    if white_ratio >= 0.65 and red_dots_ratio < 0.15:
        return "cue"

    if mean_v < 70 and white_ratio < 0.15:
        return "eight"

    white_mask = (v > 140) & (s < 75)
    non_white_hsv = hsv[~white_mask] if np.sum(~white_mask) > 0 else hsv
    hue_name = _dominant_hue_name(non_white_hsv)

    if white_ratio > 0.25:
        return f"stripe-{hue_name}"
    return f"solid-{hue_name}"


# ---------------------------------------------------------------------------
# Step 4 – Full detection pipeline + Teach Mode support
# ---------------------------------------------------------------------------

def analyse_image(
    image_bgr: np.ndarray,
    manual_cue_x: Optional[float] = None,
    manual_cue_y: Optional[float] = None,
    manual_cue_ball_id: Optional[str] = None,
) -> Optional[dict]:
    warped, H, dims = detect_table_and_warp(image_bgr)
    if warped is None or dims is None:
        return None

    pockets = build_pocket_list(dims)
    diamonds = build_diamond_list(dims)
    raw_balls = detect_balls_on_warped(warped, dims)

    px_mm = _px_per_mm_from_warped(warped, dims)
    h_px = warped.shape[0]

    candidate_balls = []

    for (x_mm, y_mm, r_mm) in raw_balls:
        cx_px = int(x_mm * px_mm)
        cy_px = int(h_px - y_mm * px_mm)
        r_px  = int(r_mm * px_mm)

        label = classify_ball(warped, cx_px, cy_px, r_px)

        x1 = max(0, cx_px - r_px)
        y1 = max(0, cy_px - r_px)
        x2 = min(warped.shape[1], cx_px + r_px)
        y2 = min(warped.shape[0], cy_px + r_px)
        roi = warped[y1:y2, x1:x2]
        w_ratio = get_ball_white_ratio(roi)

        candidate_balls.append({
            "x_mm": round(x_mm, 1),
            "y_mm": round(y_mm, 1),
            "r_mm": round(r_mm, 2),
            "label": label,
            "white_ratio": w_ratio,
        })

    # Manual Cue Ball Override / Teach Mode handling:
    cue_detected = True

    if manual_cue_ball_id is not None:
        # User selected an existing detected ball to be the cue ball
        for i, b in enumerate(candidate_balls):
            if f"obj{i+1}" == manual_cue_ball_id or manual_cue_ball_id == "cue":
                b["label"] = "cue"
    elif manual_cue_x is not None and manual_cue_y is not None:
        # User tapped coordinates to create/place cue ball
        candidate_balls = [b for b in candidate_balls if b["label"] != "cue"]
        candidate_balls.append({
            "x_mm": round(manual_cue_x, 1),
            "y_mm": round(manual_cue_y, 1),
            "r_mm": BALL_RADIUS_MM,
            "label": "cue",
            "white_ratio": 1.0,
        })
    else:
        # Automatic cue ball identification
        cue_indices = [i for i, b in enumerate(candidate_balls) if b["label"] == "cue"]
        if not cue_indices:
            # Check if any ball has high white ratio (> 0.55)
            high_white_indices = [i for i, b in enumerate(candidate_balls) if b["white_ratio"] > 0.55]
            if high_white_indices:
                best_idx = max(high_white_indices, key=lambda i: candidate_balls[i]["white_ratio"])
                candidate_balls[best_idx]["label"] = "cue"
            else:
                # Cue ball not detected! Enable Teach Mode
                cue_detected = False

    # Ensure at most one cue ball
    final_cue_indices = [i for i, b in enumerate(candidate_balls) if b["label"] == "cue"]
    if len(final_cue_indices) > 1:
        best_idx = max(final_cue_indices, key=lambda i: candidate_balls[i]["white_ratio"])
        for i in final_cue_indices:
            if i != best_idx:
                candidate_balls[i]["label"] = "stripe-red"

    # Assemble final Ball list
    balls: List[Ball] = []
    obj_counter = 0
    for b in candidate_balls:
        if b["label"] == "cue":
            ball_id = "cue"
        else:
            obj_counter += 1
            ball_id = f"obj{obj_counter}"

        balls.append(Ball(
            id=ball_id,
            label=b["label"],
            x=b["x_mm"],
            y=b["y_mm"],
            radius_mm=b["r_mm"],
        ))

    return {
        "dims": dims,
        "pockets": pockets,
        "diamonds": diamonds,
        "balls": balls,
        "cue_detected": cue_detected,
        "warped": warped,
    }

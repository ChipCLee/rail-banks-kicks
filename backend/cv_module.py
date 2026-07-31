"""
Computer Vision module for Rail-Kick.

Implements:
  1. Table boundary detection (green felt HSV masking → homography)
  2. Ball detection (Gaussian blur + HoughCircles)
  3. Ball classification (HSV color profiling)
  4. Perspective-corrected top-down image output

All measurements are in mm in table space.
SPEC.md §Feature 1 – Ball Position Analysis.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

from models import Ball, Pocket, TableDims

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
    Detect the largest green-felt region using HSV masking.
    Returns the 4-point approximated contour or None.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Green felt HSV range — covers typical billiard cloth
    lower1 = np.array([35, 40, 30])
    upper1 = np.array([95, 255, 220])
    mask = cv2.inRange(hsv, lower1, upper1)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Largest contour → table boundary
    largest = max(contours, key=cv2.contourArea)

    # Approximate to a quadrilateral
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

    if len(approx) != 4:
        # Fall back: use bounding rect corners
        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect)
        approx = box.reshape(-1, 1, 2).astype(np.float32)

    return approx


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 corner points as [TL, TR, BR, BL] (clockwise from top-left).
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # TL (smallest x+y)
    rect[2] = pts[np.argmax(s)]   # BR (largest x+y)

    diff = np.diff(pts, axis=1).ravel()
    rect[1] = pts[np.argmin(diff)]  # TR (smallest y-x)
    rect[3] = pts[np.argmax(diff)]  # BL (largest y-x)

    return rect


def detect_table_and_warp(
    image_bgr: np.ndarray,
    table_size: str = DEFAULT_TABLE,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[TableDims]]:
    """
    Detect table boundary and return:
      - warped: perspective-corrected top-down BGR image
      - H: homography matrix (image → table mm space)
      - dims: TableDims(width_mm, height_mm)

    Returns (None, None, None) if no table found.
    """
    config = TABLE_CONFIGS.get(table_size, TABLE_CONFIGS[DEFAULT_TABLE])
    w_mm = config["width_mm"]
    h_mm = config["height_mm"]

    # Render resolution: 4 px per mm
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
    """
    Build the six standard pocket positions in mm table space.
    Origin (0,0) = bottom-left corner.
    """
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


# ---------------------------------------------------------------------------
# Step 2 – Ball Detection
# ---------------------------------------------------------------------------

def _px_per_mm_from_warped(warped: np.ndarray, dims: TableDims) -> float:
    """Compute pixel-per-mm ratio from the warped image dimensions."""
    img_w = warped.shape[1]
    return img_w / dims.width


def detect_balls_on_warped(
    warped: np.ndarray,
    dims: TableDims,
) -> List[Tuple[float, float, float]]:
    """
    Run HoughCircles on the warped image.
    Returns list of (x_mm, y_mm, radius_mm) for each detected ball.
    """
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
            # Convert pixel → mm
            # warped image: y=0 is TOP of image → table y=height (TL corner)
            x_mm = cx_px / px_mm
            y_mm = (h_px - cy_px) / px_mm   # flip y: image top = table height
            r_mm = r_px / px_mm
            results.append((x_mm, y_mm, r_mm))
    return results


# ---------------------------------------------------------------------------
# Step 3 – Ball Classification
# ---------------------------------------------------------------------------

# HSV hue bucket names (0–360° mapped to 0–180 in OpenCV)
_HUE_BUCKETS = [
    (0,   10,  "red"),
    (10,  25,  "orange"),
    (25,  35,  "yellow"),
    (35,  85,  "green"),
    (85,  130, "blue"),
    (130, 160, "purple"),
    (160, 180, "red"),   # wraps back to red
]


def _dominant_hue_name(hsv_roi: np.ndarray) -> str:
    """Return the dominant hue bucket name from an HSV ROI."""
    hue_channel = hsv_roi[:, :, 0].ravel()
    if len(hue_channel) == 0:
        return "unknown"
    hist = np.bincount(hue_channel.astype(int), minlength=181)
    dominant_hue = int(np.argmax(hist))
    for lo, hi, name in _HUE_BUCKETS:
        if lo <= dominant_hue < hi:
            return name
    return "red"


def classify_ball(
    warped: np.ndarray,
    cx_px: int,
    cy_px: int,
    r_px: int,
) -> str:
    """
    Classify a single ball ROI by its colour profile.
    Returns label: "cue" | "eight" | "solid-<hue>" | "stripe-<hue>"
    """
    # Extract circular ROI
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
    mean_v = float(np.mean(v))
    mean_s = float(np.mean(s))

    # Cue ball: high brightness, low saturation
    if mean_v > 180 and mean_s < 50:
        return "cue"

    # 8-ball: very dark
    if mean_v < 70:
        return "eight"

    # Stripes have a prominent white region (top/bottom of ball)
    # Mask white pixels (high V, low S)
    white_mask = (v > 160) & (s < 60)
    white_ratio = float(np.sum(white_mask)) / white_mask.size

    hue_name = _dominant_hue_name(hsv[~white_mask.astype(bool)] if white_ratio > 0.1 else hsv)

    if white_ratio > 0.25:
        return f"stripe-{hue_name}"
    return f"solid-{hue_name}"


# ---------------------------------------------------------------------------
# Step 4 – Coordinate output + full detection pipeline
# ---------------------------------------------------------------------------

def analyse_image(image_bgr: np.ndarray) -> Optional[dict]:
    """
    Full pipeline:
      1. Detect table + warp
      2. Detect balls
      3. Classify balls
      4. Return dict with dims, pockets, balls, warped_image

    Returns None if no table detected.
    """
    warped, H, dims = detect_table_and_warp(image_bgr)
    if warped is None or dims is None:
        return None

    pockets = build_pocket_list(dims)
    raw_balls = detect_balls_on_warped(warped, dims)

    px_mm = _px_per_mm_from_warped(warped, dims)
    h_px = warped.shape[0]

    balls: List[Ball] = []
    obj_counter = 0
    for (x_mm, y_mm, r_mm) in raw_balls:
        # Convert mm back to pixel for ROI extraction
        cx_px = int(x_mm * px_mm)
        cy_px = int(h_px - y_mm * px_mm)
        r_px  = int(r_mm * px_mm)

        label = classify_ball(warped, cx_px, cy_px, r_px)

        if label == "cue":
            ball_id = "cue"
        else:
            obj_counter += 1
            ball_id = f"obj{obj_counter}"

        balls.append(Ball(
            id=ball_id,
            label=label,
            x=round(x_mm, 1),
            y=round(y_mm, 1),
            radius_mm=round(r_mm, 2),
        ))

    # Ensure at most one cue ball (keep the highest-brightness one if duplicates)
    cue_balls = [b for b in balls if b.id == "cue"]
    if len(cue_balls) > 1:
        # Keep the first, relabel rest as solid-white fallback
        for b in cue_balls[1:]:
            b.id = f"obj{obj_counter}"
            b.label = "solid-white"
            obj_counter += 1

    return {
        "dims": dims,
        "pockets": pockets,
        "balls": balls,
        "warped": warped,
    }

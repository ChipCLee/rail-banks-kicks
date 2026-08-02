"""
Computer Vision module for Rail-Kick.

Implements explicit CV rules:
  1. Long rails have 3 pockets (Corner, Middle Side, Corner) and 6 diamonds.
  2. Short rails have 2 pockets (Corner, Corner) and 3 diamonds.
  3. No ball is outside the table boundary.
  4. Once the table is identified, focus ONLY on the table and ignore all background.
  5. Automatically detects portrait table orientation and rotates the image 90° clockwise
     so that Long Rails are ALWAYS at the Top & Bottom and Short Rails are at the Left & Right,
     ensuring the camera photo and 2D diagram share the exact same horizontal landscape orientation.

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

def _detect_felt_contour(image_bgr: np.ndarray, felt_color: str = "auto") -> Optional[np.ndarray]:
    """
    Detect the pool felt region using user-selected or multi-color HSV masking.
    Options:
      - 'blue': Simonis Tournament Blue felt [85..135]
      - 'green': Traditional Green felt [30..90]
      - 'red': Red / Burgundy felt [0..10] | [160..180]
      - 'auto': Combined auto detection
    Returns the 4-point approximated contour or None.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    color_lower = felt_color.lower().strip() if felt_color else "auto"

    mask_green = cv2.inRange(hsv, np.array([30, 30, 30]), np.array([90, 255, 255]))
    mask_blue = cv2.inRange(hsv, np.array([85, 30, 30]), np.array([135, 255, 255]))
    mask_red1 = cv2.inRange(hsv, np.array([0, 40, 30]), np.array([10, 255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([160, 40, 30]), np.array([180, 255, 255]))
    mask_red = mask_red1 | mask_red2

    if color_lower == "blue":
        mask = mask_blue
    elif color_lower == "green":
        mask = mask_green
    elif color_lower == "red":
        mask = mask_red
    else:
        mask = mask_green | mask_blue | mask_red

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    img_area = image_bgr.shape[0] * image_bgr.shape[1]
    if cv2.contourArea(largest) < 0.15 * img_area:
        return None

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
    felt_color: str = "auto",
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[TableDims], bool]:
    """
    Detect table boundary, rotate portrait input images to standardized landscape orientation
    (Long Rails at Top & Bottom, Short Rails at Left & Right), and return perspective warped image.
    """
    contour = _detect_felt_contour(image_bgr, felt_color=felt_color)
    if contour is None:
        img_h, img_w = image_bgr.shape[:2]
        return None, None, None, img_h > img_w

    src_pts = _order_corners(contour)

    # Measure felt contour edges in original image
    top_len = np.linalg.norm(src_pts[1] - src_pts[0])
    right_len = np.linalg.norm(src_pts[2] - src_pts[1])
    bottom_len = np.linalg.norm(src_pts[2] - src_pts[3])
    left_len = np.linalg.norm(src_pts[3] - src_pts[0])

    horiz_len = (top_len + bottom_len) / 2.0
    vert_len = (left_len + right_len) / 2.0
    is_portrait_felt = vert_len > horiz_len

    if is_portrait_felt:
        # Rotate image 90° clockwise so Long Rails become Top & Bottom
        image_bgr = cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)
        contour = _detect_felt_contour(image_bgr, felt_color=felt_color)
        if contour is None:
            return None, None, None, True
        src_pts = _order_corners(contour)

    config = TABLE_CONFIGS.get(table_size, TABLE_CONFIGS[DEFAULT_TABLE])
    w_mm = config["width_mm"]   # 2540.0 mm (Long dimension)
    h_mm = config["height_mm"]  # 1270.0 mm (Short dimension)

    px_per_mm = 4.0
    w_px = int(w_mm * px_per_mm) # 10160 px
    h_px = int(h_mm * px_per_mm) # 5080 px

    # Standard landscape destination points (Long Rails Top/Bottom, Short Rails Left/Right):
    dst_pts = np.array([
        [0,    h_px],  # Photo TL -> Table (x=0, y=h) [TL]
        [w_px, h_px],  # Photo TR -> Table (x=w, y=h) [TR]
        [w_px, 0   ],  # Photo BR -> Table (x=w, y=0) [BR]
        [0,    0   ],  # Photo BL -> Table (x=0, y=0) [BL]
    ], dtype=np.float32)

    H, _ = cv2.findHomography(src_pts, dst_pts)
    if H is None:
        return None, None, None, is_portrait_felt

    warped = cv2.warpPerspective(image_bgr, H, (w_px, h_px))
    dims = TableDims(width=w_mm, height=h_mm)
    return warped, H, dims, is_portrait_felt


def build_pocket_list(dims: TableDims) -> List[Pocket]:
    """
    Rule 1 & 2:
    - Long rails (TOP y=h, BOTTOM y=0) have 3 pockets (Corner TL, Middle Side ML, Corner TR; Corner BL, Middle Side MR, Corner BR)
    - Short rails (LEFT x=0, RIGHT x=w) have 2 corner pockets (no side pocket)
    Total: 6 pockets.
    """
    w = dims.width   # 2540.0 mm (Long dimension)
    h = dims.height  # 1270.0 mm (Short dimension)
    return [
        Pocket(id="TL", x=0.0,   y=h,    radius_mm=CORNER_POCKET_RADIUS_MM),
        Pocket(id="TR", x=w,     y=h,    radius_mm=CORNER_POCKET_RADIUS_MM),
        Pocket(id="ML", x=w/2,   y=h,    radius_mm=SIDE_POCKET_RADIUS_MM), # Middle Side Pocket on TOP Long Rail
        Pocket(id="MR", x=w/2,   y=0.0,  radius_mm=SIDE_POCKET_RADIUS_MM), # Middle Side Pocket on BOTTOM Long Rail
        Pocket(id="BL", x=0.0,   y=0.0,  radius_mm=CORNER_POCKET_RADIUS_MM),
        Pocket(id="BR", x=w,     y=0.0,  radius_mm=CORNER_POCKET_RADIUS_MM),
    ]


def build_diamond_list(dims: TableDims) -> List[DiamondMarker]:
    """
    Rule 1: Long rails have 6 diamonds (3 on each side of side pocket).
    Rule 2: Short rails have 3 diamonds.
    Total: 6 (TOP) + 6 (BOTTOM) + 3 (LEFT) + 3 (RIGHT) = 18 diamonds.
    """
    w = dims.width
    h = dims.height
    diamonds: List[DiamondMarker] = []

    # Rule 1: Long rails (TOP & BOTTOM) - 6 diamonds per long rail
    seg_w = w / 8.0
    for i in [1, 2, 3, 5, 6, 7]:
        x_pos = round(i * seg_w, 1)
        num = (i * 0.5) if i <= 3 else ((8 - i) * 0.5)
        diamonds.append(DiamondMarker(rail="TOP", number=num, x=x_pos, y=h, label=f"{num} TOP"))
        diamonds.append(DiamondMarker(rail="BOTTOM", number=num, x=x_pos, y=0.0, label=f"{num} BTM"))

    # Rule 2: Short rails (LEFT & RIGHT) - 3 diamonds per short rail
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
    """
    Detect balls strictly within table boundaries.
    Rule 3: Discards any candidate ball center outside the table cushion playfield.
    Rule 4: Ignores all non-table background.
    """
    px_mm = _px_per_mm_from_warped(warped, dims)
    expected_r_px = BALL_RADIUS_MM * px_mm
    expected_area = math.pi * (expected_r_px ** 2)
    h_px, w_px = warped.shape[0], warped.shape[1]

    # Playfield cushion margin (75mm inside outer table boundary)
    margin_px = int(75.0 * px_mm)

    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    mask_felt = (
        cv2.inRange(hsv, np.array([30, 30, 30]), np.array([90, 255, 255])) |
        cv2.inRange(hsv, np.array([85, 30, 30]), np.array([135, 255, 255]))
    )
    non_felt = ~mask_felt

    # Rule 4: Zero out 100% of background outside the playfield cushions
    non_felt[:margin_px, :] = 0
    non_felt[-margin_px:, :] = 0
    non_felt[:, :margin_px] = 0
    non_felt[:, -margin_px:] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    non_felt = cv2.morphologyEx(non_felt, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(non_felt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected: List[Tuple[float, float, float]] = []

    for c in contours:
        area = cv2.contourArea(c)
        if 0.15 * expected_area <= area <= 3.0 * expected_area:
            (cx_px, cy_px), r_px = cv2.minEnclosingCircle(c)
            cx_px, cy_px, r_px = int(cx_px), int(cy_px), int(r_px)
            x_mm = cx_px / px_mm
            y_mm = (h_px - cy_px) / px_mm
            r_mm = BALL_RADIUS_MM

            # Rule 3: Strict assertion - No ball center is outside the table playfield
            if BALL_RADIUS_MM <= x_mm <= dims.width - BALL_RADIUS_MM and BALL_RADIUS_MM <= y_mm <= dims.height - BALL_RADIUS_MM:
                if not any(math.hypot(x_mm - ex, y_mm - ey) < BALL_RADIUS_MM * 1.5 for (ex, ey, _) in detected):
                    detected.append((x_mm, y_mm, r_mm))

    return detected


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


def _dominant_hue_name(hsv_pixels: np.ndarray) -> str:
    """Return dominant hue bucket name from 3D HSV ROI or 2D filtered pixel list."""
    if hsv_pixels.size == 0:
        return "red"

    if hsv_pixels.ndim == 3:
        hue_channel = hsv_pixels[:, :, 0].ravel()
    elif hsv_pixels.ndim == 2:
        hue_channel = hsv_pixels[:, 0].ravel()
    else:
        hue_channel = hsv_pixels.ravel()

    if len(hue_channel) == 0:
        return "red"

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
    white_pixels = (v > 140) & (s < 80)
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

    if white_ratio >= 0.50 and red_dots_ratio < 0.20:
        return "cue"

    if mean_v < 70 and white_ratio < 0.15:
        return "eight"

    white_mask = (v > 140) & (s < 80)
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
    felt_color: str = "auto",
) -> Optional[dict]:
    warped, H, dims, is_portrait = detect_table_and_warp(image_bgr, felt_color=felt_color)
    if warped is None or dims is None:
        return None

    pockets = build_pocket_list(dims)
    diamonds = build_diamond_list(dims)
    raw_balls = detect_balls_on_warped(warped, dims)

    px_mm = _px_per_mm_from_warped(warped, dims)
    h_px = warped.shape[0]

    candidate_balls = []

    for (x_mm, y_mm, r_mm) in raw_balls:
        # Rule 3: Enforce no ball is outside the table boundary
        if not (BALL_RADIUS_MM <= x_mm <= dims.width - BALL_RADIUS_MM and BALL_RADIUS_MM <= y_mm <= dims.height - BALL_RADIUS_MM):
            continue

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

    cue_detected = True

    if manual_cue_ball_id is not None:
        for i, b in enumerate(candidate_balls):
            if f"obj{i+1}" == manual_cue_ball_id or manual_cue_ball_id == "cue":
                b["label"] = "cue"
    elif manual_cue_x is not None and manual_cue_y is not None:
        candidate_balls = [b for b in candidate_balls if b["label"] != "cue"]
        candidate_balls.append({
            "x_mm": round(manual_cue_x, 1),
            "y_mm": round(manual_cue_y, 1),
            "r_mm": BALL_RADIUS_MM,
            "label": "cue",
            "white_ratio": 1.0,
        })
    else:
        cue_indices = [i for i, b in enumerate(candidate_balls) if b["label"] == "cue"]
        if not cue_indices:
            high_white_indices = [i for i, b in enumerate(candidate_balls) if b["white_ratio"] > 0.30]
            if high_white_indices:
                best_idx = max(high_white_indices, key=lambda i: candidate_balls[i]["white_ratio"])
                candidate_balls[best_idx]["label"] = "cue"
            else:
                cue_detected = False

    final_cue_indices = [i for i, b in enumerate(candidate_balls) if b["label"] == "cue"]
    if len(final_cue_indices) > 1:
        best_idx = max(final_cue_indices, key=lambda i: candidate_balls[i]["white_ratio"])
        for i in final_cue_indices:
            if i != best_idx:
                candidate_balls[i]["label"] = "stripe-red"

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
        "is_portrait": is_portrait,
    }

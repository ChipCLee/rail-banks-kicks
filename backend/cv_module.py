"""
YOLOv8-small computer-vision module for Rail-Kick.

A custom segmentation model identifies the table and balls. OpenCV reduces the
table mask to four corners, rectifies it to a top-down 2:1 playfield, classifies
generic object-ball crops, and maps centres into millimetres. The API always
normalizes long rails to the top and bottom of the warped image.

All measurements are in mm in table space.
SPEC.md §Feature 1 – Ball Position Analysis.
"""
from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Protocol, Tuple

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

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "weights" / "rail_kick_yolov8s_seg.pt"
DEFAULT_INFERENCE_SIZE = 1280
DEFAULT_WARP_WIDTH_PX = 2560


class ModelUnavailableError(RuntimeError):
    """Raised when the configured YOLO model cannot serve inference."""


@dataclass(frozen=True)
class YoloDetection:
    """Model-independent representation of one YOLO segmentation result."""

    label: str
    confidence: float
    xyxy: Tuple[float, float, float, float]
    polygon: Optional[np.ndarray] = None


class Detector(Protocol):
    device: str

    def predict(self, image_bgr: np.ndarray) -> List[YoloDetection]: ...


def select_inference_device(torch_module: Any = None, requested: Optional[str] = None) -> str:
    """Select CUDA, Apple MPS, or CPU, with an explicit environment override."""
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError as exc:  # pragma: no cover - dependency packaging failure
            raise ModelUnavailableError("PyTorch is not installed.") from exc

    choice = (requested or os.getenv("CV_DEVICE", "auto")).strip().lower()
    if choice not in {"auto", "cuda", "mps", "cpu"}:
        raise ModelUnavailableError("CV_DEVICE must be one of: auto, cuda, mps, cpu.")

    cuda_available = bool(torch_module.cuda.is_available())
    mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
    mps_available = bool(mps_backend and mps_backend.is_available())

    if choice == "cuda":
        if not cuda_available:
            raise ModelUnavailableError("CV_DEVICE=cuda was requested, but CUDA is unavailable.")
        return "cuda:0"
    if choice == "mps":
        if not mps_available:
            raise ModelUnavailableError("CV_DEVICE=mps was requested, but Apple MPS is unavailable.")
        return "mps"
    if choice == "cpu":
        return "cpu"
    if cuda_available:
        return "cuda:0"
    if mps_available:
        return "mps"
    return "cpu"


_LABEL_ALIASES = {
    "table": "table",
    "pool_table": "table",
    "billiards_table": "table",
    "cue": "cue_ball",
    "cue_ball": "cue_ball",
    "white_ball": "cue_ball",
    "8_ball": "eight_ball",
    "eight": "eight_ball",
    "eight_ball": "eight_ball",
    "ball": "object_ball",
    "pool_ball": "object_ball",
    "object_ball": "object_ball",
    "solid": "solid_ball",
    "solid_ball": "solid_ball",
    "stripe": "striped_ball",
    "striped": "striped_ball",
    "striped_ball": "striped_ball",
}


class YoloV8SmallDetector:
    """Thread-safe Ultralytics YOLOv8s segmentation adapter loaded once per process."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        *,
        device: Optional[str] = None,
        inference_size: Optional[int] = None,
    ) -> None:
        path = Path(model_path or os.getenv("YOLO_MODEL_PATH", str(DEFAULT_MODEL_PATH))).expanduser()
        if not path.is_file():
            raise ModelUnavailableError(
                f"Custom YOLOv8s weights were not found at '{path}'. Set YOLO_MODEL_PATH."
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - dependency packaging failure
            raise ModelUnavailableError("The ultralytics package is not installed.") from exc

        self.model_path = path
        self.device = device or select_inference_device()
        try:
            self.inference_size = inference_size or int(
                os.getenv("YOLO_IMAGE_SIZE", str(DEFAULT_INFERENCE_SIZE))
            )
            self.confidence = float(os.getenv("YOLO_CONFIDENCE", "0.25"))
            self.iou = float(os.getenv("YOLO_IOU", "0.50"))
            self._model = YOLO(str(path), task="segment")
        except Exception as exc:
            raise ModelUnavailableError(f"Could not load YOLOv8s weights '{path}': {exc}") from exc

        raw_names = (
            self._model.names.values()
            if isinstance(self._model.names, dict)
            else self._model.names
        )
        model_labels = {
            _LABEL_ALIASES.get(str(name).strip().lower().replace("-", "_").replace(" ", "_"), str(name))
            for name in raw_names
        }
        ball_labels = {"cue_ball", "eight_ball", "object_ball", "solid_ball", "striped_ball"}
        if "table" not in model_labels or not model_labels.intersection(ball_labels):
            raise ModelUnavailableError(
                "The checkpoint is not a Rail-Kick model: it must define 'table' and ball classes."
            )
        self._lock = threading.Lock()

    def predict(self, image_bgr: np.ndarray) -> List[YoloDetection]:
        try:
            with self._lock:
                results = self._model.predict(
                    source=image_bgr,
                    imgsz=self.inference_size,
                    conf=self.confidence,
                    iou=self.iou,
                    device=self.device,
                    verbose=False,
                )
        except Exception as exc:
            raise ModelUnavailableError(
                f"YOLOv8s inference failed on device '{self.device}': {exc}"
            ) from exc

        if not results:
            return []
        result = results[0]
        if result.boxes is None:
            return []

        names = result.names
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        confidences = result.boxes.conf.detach().cpu().numpy()
        polygons = result.masks.xy if result.masks is not None else []

        detections: List[YoloDetection] = []
        for index, (box, class_id, confidence) in enumerate(zip(boxes, classes, confidences)):
            raw_name = names[class_id] if isinstance(names, dict) else names[class_id]
            normalized_name = str(raw_name).strip().lower().replace("-", "_").replace(" ", "_")
            label = _LABEL_ALIASES.get(normalized_name, normalized_name)
            polygon = None
            if index < len(polygons) and len(polygons[index]) >= 3:
                polygon = np.asarray(polygons[index], dtype=np.float32)
            detections.append(
                YoloDetection(
                    label=label,
                    confidence=float(confidence),
                    xyxy=tuple(float(value) for value in box),
                    polygon=polygon,
                )
            )
        return detections


_detector: Optional[YoloV8SmallDetector] = None
_detector_lock = threading.Lock()


def get_detector() -> YoloV8SmallDetector:
    """Return the process-wide detector, initializing it exactly once."""
    global _detector
    if _detector is None:
        with _detector_lock:
            if _detector is None:
                _detector = YoloV8SmallDetector()
    return _detector


def model_status() -> dict:
    path = Path(os.getenv("YOLO_MODEL_PATH", str(DEFAULT_MODEL_PATH))).expanduser()
    return {
        "model": "yolov8s-seg",
        "weights": str(path),
        "weights_available": path.is_file(),
        "loaded": _detector is not None,
        "device": _detector.device if _detector is not None else None,
    }


# ---------------------------------------------------------------------------
# Retired HSV compatibility helpers (not used by analyse_image)
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


def _detect_long_rails_by_side_pockets(image_bgr: np.ndarray, rect: np.ndarray) -> bool:
    """
    Detect whether vertical rails (Left & Right) or horizontal rails (Top & Bottom)
    are the LONG RAILS by perspective warping the felt contour into a 1:1 top-view space
    and analyzing middle side pocket indicators on both horizontal and vertical edges.

    Returns True if vertical rails are long rails (portrait layout), False if horizontal rails are long rails.
    """
    top_len = np.linalg.norm(rect[1] - rect[0])
    right_len = np.linalg.norm(rect[2] - rect[1])
    bottom_len = np.linalg.norm(rect[2] - rect[3])
    left_len = np.linalg.norm(rect[3] - rect[0])

    w_top = max(100, int((top_len + bottom_len) / 2.0))
    h_top = max(100, int((left_len + right_len) / 2.0))

    # Top-view destination corners
    dst_corners = np.array([
        [0,     0    ],  # TL -> (0, 0)
        [w_top, 0    ],  # TR -> (w_top, 0)
        [w_top, h_top],  # BR -> (w_top, h_top)
        [0,     h_top],  # BL -> (0, h_top)
    ], dtype=np.float32)

    H, _ = cv2.findHomography(rect, dst_corners)
    if H is None:
        return (left_len + right_len) > (top_len + bottom_len)

    top_view = cv2.warpPerspective(image_bgr, H, (w_top, h_top))
    hsv_top = cv2.cvtColor(top_view, cv2.COLOR_BGR2HSV)
    v_chan = hsv_top[:, :, 2]

    # Non-felt mask for pocket cutout detection in top-view space
    mask_felt = (
        cv2.inRange(hsv_top, np.array([30, 30, 30]), np.array([90, 255, 255])) |
        cv2.inRange(hsv_top, np.array([85, 30, 30]), np.array([135, 255, 255])) |
        cv2.inRange(hsv_top, np.array([0, 40, 30]), np.array([10, 255, 255])) |
        cv2.inRange(hsv_top, np.array([160, 40, 30]), np.array([180, 255, 255]))
    )
    non_felt = ~mask_felt

    # Sample ROIs around midpoints of top-view edges
    r_w = max(10, int(w_top * 0.05))
    r_h = max(10, int(h_top * 0.05))

    top_v = v_chan[0:r_h, max(0, w_top//2 - r_w):min(w_top, w_top//2 + r_w)]
    top_nf = non_felt[0:r_h, max(0, w_top//2 - r_w):min(w_top, w_top//2 + r_w)]

    btm_v = v_chan[max(0, h_top - r_h):h_top, max(0, w_top//2 - r_w):min(w_top, w_top//2 + r_w)]
    btm_nf = non_felt[max(0, h_top - r_h):h_top, max(0, w_top//2 - r_w):min(w_top, w_top//2 + r_w)]

    left_v = v_chan[max(0, h_top//2 - r_h):min(h_top, h_top//2 + r_h), 0:r_w]
    left_nf = non_felt[max(0, h_top//2 - r_h):min(h_top, h_top//2 + r_h), 0:r_w]

    right_v = v_chan[max(0, h_top//2 - r_h):min(h_top, h_top//2 + r_h), max(0, w_top - r_w):w_top]
    right_nf = non_felt[max(0, h_top//2 - r_h):min(h_top, h_top//2 + r_h), max(0, w_top - r_w):w_top]

    def calc_score(v_roi, nf_roi):
        if v_roi.size == 0:
            return 0.0
        dark_ratio = float(np.sum(v_roi < 85)) / v_roi.size
        nf_ratio = float(np.sum(nf_roi > 0)) / nf_roi.size
        return 0.6 * dark_ratio + 0.4 * nf_ratio

    s_top = calc_score(top_v, top_nf)
    s_btm = calc_score(btm_v, btm_nf)
    s_left = calc_score(left_v, left_nf)
    s_right = calc_score(right_v, right_nf)

    horiz_score = s_top + s_btm
    vert_score = s_left + s_right

    return bool(vert_score > horiz_score)







def detect_table_and_warp(
    image_bgr: np.ndarray,
    table_size: str = DEFAULT_TABLE,
    felt_color: str = "auto",
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[TableDims], bool]:
    """
    Detect table boundary, use Middle Side Pocket detection (3 pockets on long rails,
    2 pockets on short rails) to rotate portrait input images 90° clockwise so that
    Long Rails are ALWAYS at Top & Bottom and Short Rails are at Left & Right.
    """
    contour = _detect_felt_contour(image_bgr, felt_color=felt_color)
    if contour is None:
        img_h, img_w = image_bgr.shape[:2]
        return None, None, None, img_h > img_w

    src_pts = _order_corners(contour)
    is_portrait_felt = _detect_long_rails_by_side_pockets(image_bgr, src_pts)

    if is_portrait_felt:
        # Rotate image 90° clockwise so Long Rails (with middle side pockets) become Top & Bottom
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
        cv2.inRange(hsv, np.array([30, 30, 30]), np.array([90, 255, 255])) |   # green
        cv2.inRange(hsv, np.array([85, 30, 30]), np.array([135, 255, 255])) |  # blue
        cv2.inRange(hsv, np.array([0, 40, 30]), np.array([10, 255, 255])) |    # red1
        cv2.inRange(hsv, np.array([160, 40, 30]), np.array([180, 255, 255]))   # red2
    )
    non_felt = ~mask_felt

    # Rule 4: Zero out 100% of background outside the playfield cushions
    non_felt[:margin_px, :] = 0
    non_felt[-margin_px:, :] = 0
    non_felt[:, :margin_px] = 0
    non_felt[:, -margin_px:] = 0

    # Pocket exclusion zones — mask out circular regions around all 6 pocket locations
    # so dark pocket holes are never considered as ball candidates
    pockets = build_pocket_list(dims)
    for pocket in pockets:
        pocket_cx = int(pocket.x * px_mm)
        pocket_cy = int(h_px - pocket.y * px_mm)
        # Use 2× pocket radius for generous exclusion to account for perspective warp artifacts
        pocket_excl_r = int(pocket.radius_mm * px_mm * 2.0)
        cv2.circle(non_felt, (pocket_cx, pocket_cy), pocket_excl_r, 0, -1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    non_felt = cv2.morphologyEx(non_felt, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(non_felt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected: List[Tuple[float, float, float]] = []

    for c in contours:
        area = cv2.contourArea(c)
        if 0.15 * expected_area <= area <= 3.0 * expected_area:
            # Circularity check: reject highly irregular contours (rail edges, artifacts)
            # Perfect circle = 1.0; threshold 0.30 is lenient since pocket exclusion
            # zones are the primary defense against pocket mis-detection
            perimeter = cv2.arcLength(c, True)
            if perimeter > 0:
                circularity = (4.0 * math.pi * area) / (perimeter * perimeter)
            else:
                circularity = 0.0
            if circularity < 0.30:
                continue

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
        # Distinguish real eight balls from dark pocket holes:
        # Real balls have a specular (glossy) highlight from ambient lighting;
        # pocket holes are uniformly dark with no bright spots.
        bright_spot_mask = (v > 180)
        gloss_ratio = float(np.sum(bright_spot_mask)) / bright_spot_mask.size
        if gloss_ratio > 0.01:
            return "eight"
        # No glossy highlight detected — likely a pocket hole, not a ball
        return "unknown"

    white_mask = (v > 140) & (s < 80)
    non_white_hsv = hsv[~white_mask] if np.sum(~white_mask) > 0 else hsv
    hue_name = _dominant_hue_name(non_white_hsv)

    if white_ratio > 0.25:
        return f"stripe-{hue_name}"
    return f"solid-{hue_name}"


# ---------------------------------------------------------------------------
# Step 4 – Full detection pipeline + Teach Mode support
# ---------------------------------------------------------------------------

def _analyse_image_legacy(
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

        # Skip unclassifiable objects (e.g. pocket holes that slipped through)
        if label == "unknown":
            continue

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


# ---------------------------------------------------------------------------
# YOLOv8s production pipeline
# ---------------------------------------------------------------------------

def _polygon_to_quad(polygon: np.ndarray) -> Optional[np.ndarray]:
    """Reduce a table segmentation polygon to perspective corners [TL, TR, BR, BL]."""
    if polygon is None or len(polygon) < 4:
        return None
    contour = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    for epsilon in np.linspace(0.005, 0.08, 16):
        approximation = cv2.approxPolyDP(hull, float(epsilon) * perimeter, True)
        if len(approximation) == 4:
            return _order_corners(approximation)

    # A mask with rounded pocket cut-outs may never simplify to four points.
    points = hull.reshape(-1, 2)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).ravel()
    quad = np.array(
        [
            points[np.argmin(sums)],
            points[np.argmin(differences)],
            points[np.argmax(sums)],
            points[np.argmax(differences)],
        ],
        dtype=np.float32,
    )
    if len(np.unique(quad, axis=0)) != 4:
        return None
    return quad


def _yolo_table_warp(
    image_bgr: np.ndarray,
    detections: List[YoloDetection],
    table_size: str = DEFAULT_TABLE,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[TableDims], bool]:
    table_detections = [d for d in detections if d.label == "table" and d.polygon is not None]
    if not table_detections:
        return None, None, None, image_bgr.shape[0] > image_bgr.shape[1]

    def table_score(detection: YoloDetection) -> float:
        area = abs(cv2.contourArea(detection.polygon.reshape(-1, 1, 2)))
        return area * detection.confidence

    table = max(table_detections, key=table_score)
    corners = _polygon_to_quad(table.polygon)
    if corners is None:
        return None, None, None, image_bgr.shape[0] > image_bgr.shape[1]

    top = np.linalg.norm(corners[1] - corners[0])
    bottom = np.linalg.norm(corners[2] - corners[3])
    left = np.linalg.norm(corners[3] - corners[0])
    right = np.linalg.norm(corners[2] - corners[1])
    is_portrait = bool((left + right) > (top + bottom))

    # Rotate corner correspondence clockwise when the long rails are vertical.
    source = corners[[3, 0, 1, 2]] if is_portrait else corners
    config = TABLE_CONFIGS.get(table_size, TABLE_CONFIGS[DEFAULT_TABLE])
    dims = TableDims(width=config["width_mm"], height=config["height_mm"])
    width_px = int(os.getenv("CV_WARP_WIDTH", str(DEFAULT_WARP_WIDTH_PX)))
    height_px = max(1, round(width_px * dims.height / dims.width))
    destination = np.array(
        [[0, 0], [width_px - 1, 0], [width_px - 1, height_px - 1], [0, height_px - 1]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(source.astype(np.float32), destination)
    warped = cv2.warpPerspective(image_bgr, homography, (width_px, height_px))
    return warped, homography, dims, is_portrait


def _detection_center(detection: YoloDetection) -> Tuple[float, float]:
    if detection.polygon is not None and len(detection.polygon) >= 3:
        moments = cv2.moments(detection.polygon.astype(np.float32))
        if moments["m00"]:
            return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
    x1, y1, x2, y2 = detection.xyxy
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _classify_yolo_ball(warped: np.ndarray, detection: YoloDetection) -> Tuple[str, float]:
    x1, y1, x2, y2 = detection.xyxy
    cx, cy = _detection_center(detection)
    radius = max(2, round(max(x2 - x1, y2 - y1) / 2.0))
    visual_label = classify_ball(warped, round(cx), round(cy), radius)

    if detection.label == "cue_ball":
        label = "cue"
    elif detection.label == "eight_ball":
        label = "eight"
    elif detection.label == "solid_ball":
        label = visual_label if visual_label.startswith("solid-") else "solid-red"
    elif detection.label == "striped_ball":
        label = visual_label if visual_label.startswith("stripe-") else "stripe-red"
    else:
        label = visual_label

    roi = warped[
        max(0, round(y1)):min(warped.shape[0], round(y2)),
        max(0, round(x1)):min(warped.shape[1], round(x2)),
    ]
    return label, get_ball_white_ratio(roi)


def _yolo_balls(
    warped: np.ndarray,
    dims: TableDims,
    detections: List[YoloDetection],
) -> List[dict]:
    accepted = {"cue_ball", "eight_ball", "object_ball", "solid_ball", "striped_ball"}
    width_denominator = max(1, warped.shape[1] - 1)
    height_denominator = max(1, warped.shape[0] - 1)
    candidates: List[dict] = []

    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        if detection.label not in accepted:
            continue
        cx, cy = _detection_center(detection)
        x_mm = cx / width_denominator * dims.width
        y_mm = dims.height - (cy / height_denominator * dims.height)
        if not (
            BALL_RADIUS_MM <= x_mm <= dims.width - BALL_RADIUS_MM
            and BALL_RADIUS_MM <= y_mm <= dims.height - BALL_RADIUS_MM
        ):
            continue
        if any(
            math.hypot(x_mm - existing["x_mm"], y_mm - existing["y_mm"])
            < BALL_RADIUS_MM * 1.5
            for existing in candidates
        ):
            continue

        label, white_ratio = _classify_yolo_ball(warped, detection)
        if label == "unknown":
            continue
        candidates.append(
            {
                "x_mm": round(x_mm, 1),
                "y_mm": round(y_mm, 1),
                "r_mm": BALL_RADIUS_MM,
                "label": label,
                "white_ratio": white_ratio,
                "confidence": detection.confidence,
            }
        )
    return candidates


def analyse_image(
    image_bgr: np.ndarray,
    manual_cue_x: Optional[float] = None,
    manual_cue_y: Optional[float] = None,
    manual_cue_ball_id: Optional[str] = None,
    felt_color: str = "auto",
    detector: Optional[Detector] = None,
) -> Optional[dict]:
    """Analyze a camera image with custom YOLOv8s segmentation predictions.

    ``felt_color`` remains accepted for API compatibility but learned table
    segmentation intentionally replaces color-threshold table selection.
    """
    del felt_color
    active_detector = detector or get_detector()
    table_predictions = active_detector.predict(image_bgr)
    warped, _, dims, is_portrait = _yolo_table_warp(image_bgr, table_predictions)
    if warped is None or dims is None:
        return None

    ball_predictions = active_detector.predict(warped)
    candidate_balls = _yolo_balls(warped, dims, ball_predictions)

    if manual_cue_ball_id is not None:
        for index, ball in enumerate(candidate_balls):
            if manual_cue_ball_id in {f"obj{index + 1}", "cue"}:
                ball["label"] = "cue"
                break
    elif manual_cue_x is not None and manual_cue_y is not None:
        candidate_balls = [ball for ball in candidate_balls if ball["label"] != "cue"]
        if (
            BALL_RADIUS_MM <= manual_cue_x <= dims.width - BALL_RADIUS_MM
            and BALL_RADIUS_MM <= manual_cue_y <= dims.height - BALL_RADIUS_MM
        ):
            candidate_balls.append(
                {
                    "x_mm": round(manual_cue_x, 1),
                    "y_mm": round(manual_cue_y, 1),
                    "r_mm": BALL_RADIUS_MM,
                    "label": "cue",
                    "white_ratio": 1.0,
                    "confidence": 1.0,
                }
            )

    cue_indices = [i for i, ball in enumerate(candidate_balls) if ball["label"] == "cue"]
    if not cue_indices:
        white_candidates = [i for i, ball in enumerate(candidate_balls) if ball["white_ratio"] > 0.30]
        if white_candidates:
            best = max(
                white_candidates,
                key=lambda i: (candidate_balls[i]["white_ratio"], candidate_balls[i]["confidence"]),
            )
            candidate_balls[best]["label"] = "cue"
            cue_indices = [best]
    elif len(cue_indices) > 1:
        best = max(cue_indices, key=lambda i: candidate_balls[i]["confidence"])
        for index in cue_indices:
            if index != best:
                candidate_balls[index]["label"] = "stripe-red"
        cue_indices = [best]

    balls: List[Ball] = []
    object_counter = 0
    for candidate in candidate_balls:
        if candidate["label"] == "cue":
            ball_id = "cue"
        else:
            object_counter += 1
            ball_id = f"obj{object_counter}"
        balls.append(
            Ball(
                id=ball_id,
                label=candidate["label"],
                x=candidate["x_mm"],
                y=candidate["y_mm"],
                radius_mm=candidate["r_mm"],
            )
        )

    return {
        "dims": dims,
        "pockets": build_pocket_list(dims),
        "diamonds": build_diamond_list(dims),
        "balls": balls,
        "cue_detected": bool(cue_indices),
        "warped": warped,
        "is_portrait": is_portrait,
    }

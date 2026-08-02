"""Deterministic YOLO-shaped test doubles shared by unit and E2E tests."""
from __future__ import annotations

from typing import List

import numpy as np

from cv_module import DEFAULT_WARP_WIDTH_PX, YoloDetection


class FakePoolDetector:
    device = "cpu:test"

    def __init__(self, *, portrait: bool = False, include_table: bool = True) -> None:
        self.portrait = portrait
        self.include_table = include_table
        self.calls = 0

    def predict(self, image_bgr: np.ndarray) -> List[YoloDetection]:
        self.calls += 1
        height, width = image_bgr.shape[:2]
        if width == DEFAULT_WARP_WIDTH_PX:
            return [
                YoloDetection("cue_ball", 0.98, (460, 760, 540, 840)),
                YoloDetection("eight_ball", 0.96, (1240, 600, 1320, 680)),
                YoloDetection("solid_ball", 0.92, (1960, 350, 2040, 430)),
            ]
        if not self.include_table:
            return []

        if self.portrait:
            polygon = np.array(
                [[width * 0.35, height * 0.08], [width * 0.65, height * 0.08],
                 [width * 0.70, height * 0.92], [width * 0.30, height * 0.92]],
                dtype=np.float32,
            )
        else:
            polygon = np.array(
                [[width * 0.08, height * 0.25], [width * 0.92, height * 0.20],
                 [width * 0.88, height * 0.80], [width * 0.12, height * 0.85]],
                dtype=np.float32,
            )
        return [
            YoloDetection(
                label="table",
                confidence=0.99,
                xyxy=(0.0, 0.0, float(width), float(height)),
                polygon=polygon,
            )
        ]

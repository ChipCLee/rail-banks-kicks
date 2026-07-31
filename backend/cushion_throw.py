"""
v2 Feature 4: Cushion Throw Modelling & Empirical Correction.

Implements SPEC.md §Feature 4:
  - Friction between ball and cushion causes rebound angle to be slightly shallower than geometric.
  - Formula: throw_correction_deg = k * cos(2 * bank_angle_rad), max k = 5.0 degrees.
  - Rebound angle: adjusted_rebound_angle_deg = bank_angle_deg - throw_correction_deg.
"""
from __future__ import annotations

import math
from typing import Tuple

MAX_THROW_K_DEG = 5.0


def calculate_cushion_throw(bank_angle_deg: float) -> Tuple[float, float]:
    """
    Calculate cushion throw correction and adjusted rebound angle in degrees.
    
    Returns:
      (throw_correction_deg, adjusted_rebound_angle_deg)
    """
    angle_rad = math.radians(bank_angle_deg)
    # Cosine model: maximum throw at shallow angles (0°, 45°), zero at 90°
    throw_corr = MAX_THROW_K_DEG * abs(math.cos(angle_rad))
    # At 90° (perpendicular), throw is 0
    if abs(bank_angle_deg - 90.0) < 1e-3:
        throw_corr = 0.0

    adjusted_angle = bank_angle_deg - throw_corr
    return (round(throw_corr, 2), round(adjusted_angle, 2))

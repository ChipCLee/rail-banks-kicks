"""
Pydantic data models for Rail-Kick API.
Matches the TypeScript interfaces defined in SPEC.md §Data Models & §v2 Scope.
"""
from __future__ import annotations
from typing import Literal, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

class Point(BaseModel):
    x: float
    y: float


class TableDims(BaseModel):
    width: float   # mm
    height: float  # mm


# ---------------------------------------------------------------------------
# Ball
# ---------------------------------------------------------------------------

class Ball(BaseModel):
    id: str
    label: str          # "cue" | "solid-<hue>" | "stripe-<hue>" | "eight"
    x: float            # mm from bottom-left
    y: float            # mm from bottom-left
    radius_mm: float = Field(default=28.575)


# ---------------------------------------------------------------------------
# Pocket
# ---------------------------------------------------------------------------

PocketId = Literal["TL", "TR", "ML", "MR", "BL", "BR"]

class Pocket(BaseModel):
    id: PocketId
    x: float
    y: float
    radius_mm: float    # corner ≈ 57 mm, side ≈ 63 mm


# ---------------------------------------------------------------------------
# Diamond Marker
# ---------------------------------------------------------------------------

class DiamondMarker(BaseModel):
    rail: Literal["TOP", "BOTTOM", "LEFT", "RIGHT"]
    number: float       # 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5 ...
    x: float            # mm
    y: float            # mm
    label: str          # e.g. "2.5 TOP"


# ---------------------------------------------------------------------------
# Shots
# ---------------------------------------------------------------------------

RailId = Literal["TOP", "BOTTOM", "LEFT", "RIGHT"]


class DirectShot(BaseModel):
    shot_type: Literal["direct"] = "direct"
    cue_ball: Point
    object_ball_id: str
    object_ball_label: str
    path: List[Point]           # [cue, object, pocket]
    ease_score: float = 0.0     # always 0 for direct shots
    pocket_id: PocketId


class BankShot(BaseModel):
    shot_type: Literal["one_bank"] = "one_bank"
    cue_ball: Point
    object_ball_id: str
    object_ball_label: str
    rail: RailId
    contact_point: Point        # where object ball hits the rail
    path: List[Point]           # [cue, object, rail_contact, pocket]
    bank_angle_deg: float       # angle of incidence at the rail
    throw_correction_deg: Optional[float] = None
    adjusted_rebound_angle_deg: Optional[float] = None
    ease_score: float           # |bank_angle_deg - 90| — lower is easier
    pocket_id: PocketId


class KickShot(BaseModel):
    shot_type: Literal["one_rail_kick"] = "one_rail_kick"
    cue_ball: Point
    object_ball_id: str
    object_ball_label: str
    rail: RailId
    contact_point: Point        # where cue ball hits the rail
    diamond_label: str          # e.g. "2.5 diamonds from TL on TOP rail"
    path: List[Point]           # [cue, rail_contact, object, pocket]
    bank_angle_deg: float
    throw_correction_deg: Optional[float] = None
    ease_score: float
    pocket_id: PocketId


# ---------------------------------------------------------------------------
# Analysis Result
# ---------------------------------------------------------------------------

class AnalysisResult(BaseModel):
    table_dims_mm: TableDims
    pockets: List[Pocket]
    diamonds: List[DiamondMarker] = Field(default_factory=list)
    balls: List[Ball]
    cue_detected: bool = True
    direct_shots: List[DirectShot] = Field(default_factory=list)
    bank_shots: List[BankShot] = Field(default_factory=list)
    kick_shots: List[KickShot] = Field(default_factory=list)
    cv_diagram_b64: str = ""         # clean 2D CV detection diagram (balls, pockets, diamonds)
    annotated_image_b64: str         # base64-encoded JPEG with shot overlays


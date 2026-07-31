"""
Pydantic data models for Rail-Kick API.
Matches the TypeScript interfaces defined in SPEC.md §Data Models.
"""
from __future__ import annotations
from typing import Literal, List
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
    ease_score: float           # |bank_angle_deg - 90| — lower is easier
    pocket_id: PocketId


# ---------------------------------------------------------------------------
# Analysis Result
# ---------------------------------------------------------------------------

class AnalysisResult(BaseModel):
    table_dims_mm: TableDims
    pockets: List[Pocket]
    balls: List[Ball]
    direct_shots: List[DirectShot]   # sorted by ease_score ascending
    bank_shots: List[BankShot]       # sorted by ease_score ascending
    annotated_image_b64: str         # base64-encoded JPEG of annotated top-down image

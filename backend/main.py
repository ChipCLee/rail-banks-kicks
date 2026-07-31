"""
FastAPI application for Rail-Kick pool table shot analysis.

Provides REST endpoint:
  POST /analyze  - Upload pool table image (with optional manual cue ball override),
                   returns AnalysisResult JSON with clean 2D CV Detection Diagram and shot overlays.
  GET /health    - Healthcheck endpoint
"""
from __future__ import annotations

import io
from typing import Optional
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from PIL import Image

from models import AnalysisResult
from cv_module import analyse_image
from geometry import find_direct_shots, find_bank_shots
from v2_kick_shots import find_kick_shots
from annotate import annotate_table, render_2d_cv_diagram

app = FastAPI(
    title="Rail-Kick API",
    description="Pool table shot detection backend API with 2D CV Diagram & Teach Mode",
    version="0.6.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB


@app.get("/health")
def healthcheck():
    return {"status": "ok", "version": "0.6.0"}


@app.post("/analyze", response_model=AnalysisResult)
async def analyze_table_image(
    image: UploadFile = File(...),
    manual_cue_x: Optional[float] = Form(None),
    manual_cue_y: Optional[float] = Form(None),
    manual_cue_ball_id: Optional[str] = Form(None),
):
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported image format '{image.content_type}'. Allowed: JPG, PNG, WEBP.",
        )

    content = await image.read()
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds 20 MB limit.",
        )

    try:
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
        img_np = np.array(pil_img)[:, :, ::-1]  # RGB to BGR
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image file: {str(err)}",
        )

    cv_res = analyse_image(
        img_np,
        manual_cue_x=manual_cue_x,
        manual_cue_y=manual_cue_y,
        manual_cue_ball_id=manual_cue_ball_id,
    )
    if cv_res is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Could not detect pool table boundary in image.",
        )

    dims = cv_res["dims"]
    pockets = cv_res["pockets"]
    diamonds = cv_res["diamonds"]
    balls = cv_res["balls"]
    warped = cv_res["warped"]
    cue_detected = cv_res["cue_detected"]
    is_portrait = cv_res.get("is_portrait", False)

    # 1. Render clean 2D CV Detection Diagram matching original image orientation
    cv_diagram_b64 = render_2d_cv_diagram(warped, dims, pockets, balls, diamonds, is_portrait=is_portrait)

    cue_ball = next((b for b in balls if b.id == "cue"), None)

    if not cue_detected or cue_ball is None:
        # Teach Mode: Cue ball not detected automatically
        b64_img = annotate_table(
            warped,
            dims,
            pockets,
            balls,
            direct_shots=[],
            bank_shots=[],
            kick_shots=[],
            diamonds=diamonds,
            selected_shot_index=None,
            is_portrait=is_portrait,
        )
        return AnalysisResult(
            table_dims_mm=dims,
            pockets=pockets,
            diamonds=diamonds,
            balls=balls,
            cue_detected=False,
            direct_shots=[],
            bank_shots=[],
            kick_shots=[],
            cv_diagram_b64=cv_diagram_b64,
            annotated_image_b64=b64_img,
        )

    object_balls = [b for b in balls if b.id != "cue"]

    # Detect shots
    direct_shots = find_direct_shots(cue_ball, object_balls, pockets, balls)
    bank_shots = find_bank_shots(cue_ball, object_balls, pockets, balls, dims.width, dims.height)
    kick_shots = find_kick_shots(cue_ball, object_balls, pockets, balls, dims.width, dims.height)

    # Annotate image with active shot trajectories matching original image orientation
    b64_img = annotate_table(
        warped,
        dims,
        pockets,
        balls,
        direct_shots,
        bank_shots,
        kick_shots,
        diamonds=diamonds,
        selected_shot_index=0,
        is_portrait=is_portrait,
    )

    return AnalysisResult(
        table_dims_mm=dims,
        pockets=pockets,
        diamonds=diamonds,
        balls=balls,
        cue_detected=True,
        direct_shots=direct_shots,
        bank_shots=bank_shots,
        kick_shots=kick_shots,
        cv_diagram_b64=cv_diagram_b64,
        annotated_image_b64=b64_img,
    )

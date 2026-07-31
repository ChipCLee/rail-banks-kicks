# PR: Feature v1 — Direct Shot & One-Bank Shot Detection Engine & Web SPA

## Description
This pull request implements the **v1 core features** of the Rail-Kick Pool Table Shot Detection System as specified in `SPEC.md` §1 & §2.

## Key Changes
1. **Python FastAPI Backend** (`backend/`):
   - `models.py`: Pydantic data schemas for `Ball`, `Pocket`, `DirectShot`, `BankShot`, and `AnalysisResult`.
   - `cv_module.py`: Automatic table boundary detection via green felt HSV masking and homography perspective warp; HoughCircles ball detection & color classification (`cue`, `eight`, `solid-<hue>`, `stripe-<hue>`).
   - `geometry.py`: Line segment perpendicular distance obstruction check (ghost-ball clearance, 57.15mm threshold), direct shot detection, and one-bank reflection shot detection off all 4 rails excluding pocket opening zones.
   - `annotate.py`: Renders top-down perspective-corrected visualization overlay (blue solid arrow for cue→obj, orange dashed arrow for obj→rail, green dashed arrow for rail→pocket, yellow target highlight, purple target pocket ring).
   - `main.py`: `POST /analyze` REST endpoint with 20 MB file size limit and format validation.

2. **React SPA Frontend** (`frontend/`):
   - Modern dark-mode UI with Vite + React + Vanilla CSS.
   - Mobile-first ResultScreen layout (annotated image full-width on top, scrollable shot list below).
   - ShotList grouped into "Direct Shots" and "Bank Shots", ranked by ease score (`|angle - 90|`).
   - Interactive tap-to-highlight shot path updating top-down overlay.
   - Friendly empty/error state when no valid shots exist or table boundary is unreadable.

3. **DevOps & Verification**:
   - `docker-compose.yml`, `backend/Dockerfile`, `frontend/Dockerfile`.
   - `test_geometry.py`: Unit test suite covering geometry algorithms (100% pass).

## Verification
- Unit test suite: `python3 test_geometry.py` → 4/4 passed.
- Pushed to `feature/v1-backend` on `origin`.

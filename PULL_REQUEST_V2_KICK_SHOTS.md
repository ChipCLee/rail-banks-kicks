# PR: Feature v2 — One-Rail Kick Shot Detection & Diamond System

## Description
This pull request implements **v2 Feature 3: One-Rail Kick Shot Detection & Diamond System** as specified in `SPEC.md` §3.

## Key Changes
1. **Kick Shot Detection Engine** (`backend/v2_kick_shots.py`):
   - Computes mirror reflection of object ball across all four rails.
   - Determines exact cue ball rail contact point $P$.
   - Converts contact points to standard 0–4 diamond units per half-rail (e.g., `"2.5 diamonds from TL on TOP rail"`).
   - Verifies clear paths for Cue → Rail, Rail → Object, and Object → Pocket using full 57.15 mm ghost-ball clearance tests.

2. **Visual Overlays & Data Models**:
   - `KickShot` Pydantic model with `shot_type: "one_rail_kick"` and `diamond_label`.
   - `annotate.py`: Draws blue solid arrow (Cue → Rail), orange solid arrow (Rail → Object), green dashed arrow (Object → Pocket), and diamond marker icon 🔷 on the rail cushion.

3. **Frontend Integration**:
   - `ShotList.jsx` displays a new "Kick Shots" group with diamond markers and pocket targets.

4. **Tests & Verification**:
   - `test_v2_kick.py`: Unit tests validating 2.5-diamond calculation and 8-ball kick shot worked example from SPEC §3.1 (100% pass).

## Verification
- Unit test suite: `python3 test_v2_kick.py` → 2/2 passed.
- All previous unit tests: `python3 test_geometry.py` → 4/4 passed.
- Pushed to `feature/v2-kick-shots` on `origin`.

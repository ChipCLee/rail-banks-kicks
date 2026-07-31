# PR: Feature v2 — Cushion Throw Modelling & Empirical Correction

## Description
This pull request implements **v2 Feature 4: Cushion Throw Modelling & Empirical Correction** as specified in `SPEC.md` §4.

## Key Changes
1. **Cushion Throw Engine** (`backend/cushion_throw.py`):
   - Empirical model accounting for cushion friction at shallow angles: $\text{throw\_correction} = k \cdot |\cos(\theta)|$ with $k = 5.0^\circ$.
   - Calculates $\text{adjusted\_rebound\_angle} = \text{bank\_angle} - \text{throw\_correction}$.
   - Rebound at $90^\circ$ (square impact) correctly yields $0^\circ$ throw correction.

2. **Shot Integrations & Data Models**:
   - `BankShot` and `KickShot` schemas include optional `throw_correction_deg` and `adjusted_rebound_angle_deg` fields.
   - `ShotList.jsx` renders purple throw correction badges for shallow bank shots.

3. **Tests & Verification**:
   - `test_v2_throw.py`: Unit tests validating zero throw at 90°, cosine correction at 45°, and data model population (100% pass).
   - Entire backend test suite (`python3 -m unittest discover`): **9/9 tests passed**.

## Verification
- Unit test suite: `python3 -m unittest discover` → 9/9 passed.
- Pushed to `feature/v2-cushion-throw` on `origin`.

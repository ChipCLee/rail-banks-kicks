# Rail-Kick E2E Test Cases

> Based on [SPEC.md](./SPEC.md) v0.6  
> Scope: v1 features only (direct shots + one-bank shots)

---

## Test Suites

| Suite | Area |
|---|---|
| [TC-U] Upload UI | File upload interaction and validation |
| [TC-D] Detection | Table boundary + ball position analysis |
| [TC-S] Direct Shot | Direct pocketable shot detection |
| [TC-B] Bank Shot | One-bank shot detection |
| [TC-R] Results UI | Result screen layout, ranking, interactions |
| [TC-E] Error & Edge Cases | Invalid inputs and boundary conditions |

---

## TC-U — Upload UI

### TC-U-01 · File picker upload (JPG)

**Scenario**: User uploads a valid JPG pool table photo via the file picker button.

**Preconditions**: App is on the Upload Screen. A valid 9-foot pool table JPG photo is available (< 20 MB, full table visible, overhead angle).

**Steps**:
1. Open the web app in a browser.
2. Click the "Choose File" button.
3. Select a valid JPG file.
4. Observe the screen transition.

**Expected**:
- Processing Screen appears immediately after selection.
- A spinner / progress indicator is visible.
- The app sends a `POST /analyze` request with `multipart/form-data` containing the image.

---

### TC-U-02 · Drag-and-drop upload (PNG)

**Scenario**: User drags and drops a valid PNG file onto the drop zone.

**Preconditions**: App is on the Upload Screen. A valid PNG pool table image is available.

**Steps**:
1. Open the web app.
2. Drag a PNG file from the file system and drop it onto the full-page drop zone.

**Expected**:
- Drop zone highlights on hover (visual feedback).
- Processing Screen appears after drop.
- `POST /analyze` is sent with the PNG file.

---

### TC-U-03 · Drag-and-drop upload (WEBP)

**Scenario**: User uploads a WEBP file.

**Steps**:
1. Drag and drop a valid WEBP file onto the drop zone.

**Expected**: Same behaviour as TC-U-02. WEBP is accepted.

---

### TC-U-04 · Rejected file format (PDF)

**Scenario**: User attempts to upload a PDF.

**Preconditions**: A PDF file is available.

**Steps**:
1. Click "Choose File" and select a `.pdf` file.
   OR drag and drop a PDF onto the drop zone.

**Expected**:
- File is **rejected** before any network request is sent.
- An inline error is shown: format not accepted (JPG, PNG, WEBP only).
- App remains on the Upload Screen.
- No `POST /analyze` request is made.

---

### TC-U-05 · File size exceeds 20 MB

**Scenario**: User uploads an image larger than 20 MB.

**Steps**:
1. Attempt to upload an image file > 20 MB.

**Expected**:
- File is **rejected** client-side before upload.
- Error message shown: file size limit exceeded.
- App stays on Upload Screen.

---

### TC-U-06 · Processing spinner visible during analysis

**Scenario**: Verify the Processing Screen is shown while the backend is working.

**Preconditions**: Backend response is artificially delayed (or use a slow network).

**Steps**:
1. Upload a valid image.
2. Observe the screen immediately after upload.

---

### TC-U-07 · Felt Color Selection & Table Detection Accuracy

**Scenario**: User selects table felt color (Blue, Green, Red, or Auto) before uploading a photo.

**Preconditions**: App is on the Upload Screen. Photos of Simonis Blue, Traditional Green, or Red/Burgundy pool tables are available.

**Steps**:
1. Open the web app.
2. Click the "Simonis Blue" felt color pill on the Upload Screen.
3. Select and upload a Simonis 860 Tournament Blue table photo.

**Expected**:
- The selected felt color pill (`blue`) is visually highlighted.
- `POST /analyze` request contains `felt_color=blue` in `multipart/form-data`.
- Backend uses targeted blue felt HSV masking to detect the table contour accurately.


**Expected**:
- Processing Screen is shown with a visible spinner.
- Upload button / drop zone is not shown during processing.
- User cannot re-upload while processing is in progress.

---

## TC-D — Table & Ball Detection

### TC-D-01 · Full table boundary detection

**Scenario**: Backend correctly identifies the four table corners and six pocket centres.

**Preconditions**: Photo of a standard 9-foot pool table taken from overhead. All four corners are visible.

**Steps**:
1. Upload the photo.
2. Inspect the `AnalysisResult` JSON returned by `POST /analyze`.

**Expected**:
- `pockets` array contains exactly **6 entries**: `TL`, `TR`, `ML`, `MR`, `BL`, `BR`.
- Each pocket has `x` and `y` in mm within the expected playfield dimensions (`width ≤ 2540`, `height ≤ 1270`).
- `table_dims_mm` is present with positive `width` and `height`.

---

### TC-D-02 · Cue ball identified with label "cue"

**Scenario**: The white cue ball is correctly classified.

**Preconditions**: Table photo contains a clearly visible white cue ball.

**Steps**:
1. Upload photo.
2. Inspect `balls` array in response.

**Expected**:
- Exactly one ball has `label: "cue"`.
- Its `x`, `y` are within the table bounds.
- `radius_mm` is approximately `28.575` (±2 mm tolerance).

---

### TC-D-03 · Solid ball classified as "solid-\<hue\>"

**Scenario**: A solid-coloured ball (e.g. the red 3-ball) is classified correctly.

**Steps**:
1. Upload a photo containing at least one solid ball.
2. Inspect the `balls` array.

**Expected**:
- At least one ball has `label` matching the pattern `"solid-<hue>"` (e.g. `"solid-red"`, `"solid-yellow"`).
- No `number` field is present (no OCR).

---

### TC-D-04 · Stripe ball classified as "stripe-\<hue\>"

**Scenario**: A striped ball is classified correctly.

**Steps**:
1. Upload a photo containing at least one striped ball.
2. Inspect `balls` array.

**Expected**:
- At least one ball has `label` matching `"stripe-<hue>"` (e.g. `"stripe-blue"`, `"stripe-orange"`).

---

### TC-D-05 · 8-ball classified as "eight"

**Scenario**: The black 8-ball is classified correctly.

**Steps**:
1. Upload a photo with the 8-ball clearly visible.
2. Inspect `balls` array.

**Expected**:
- Exactly one ball (or zero if not on table) has `label: "eight"`.

---

### TC-D-06 · No ball number recognition

**Scenario**: Verify OCR is not applied — no ball number data in response.

**Steps**:
1. Upload any valid table photo with numbered balls.
2. Inspect each item in the `balls` array.

**Expected**:
- No ball object has a `number` property.
- Labels are color/type strings only.

---

### TC-D-07 · Coordinate mapping sanity check

**Scenario**: Ball coordinates are in table mm space, not raw pixels.

**Steps**:
1. Upload photo of a 9-foot table with a ball placed at an identifiable position.
2. Inspect the ball's `x` and `y` values.

**Expected**:
- All ball `x` values are in `[0, 2540]` and `y` values in `[0, 1270]`.
- Pixel values such as `1920` (full-width pixel) are **not** returned.

---

## TC-S — Direct Shot Detection

### TC-S-01 · Clear direct shot detected (object ball inline with pocket)

**Scenario**: An object ball is positioned directly between the cue ball and a pocket with no obstructions.

**Preconditions**: Synthetic or real table position where the geometry guarantees a direct shot exists.

**Steps**:
1. Upload the photo.
2. Inspect `direct_shots` in response.

**Expected**:
- `direct_shots` contains at least one entry.
- The entry has `shot_type: "direct"`.
- `pocket_id` is a valid pocket ID (`TL`, `TR`, `ML`, `MR`, `BL`, or `BR`).
- `path` has 3 waypoints: `[cue_ball_centre, object_ball_centre, pocket_centre]`.
- `ease_score` is `0`.

---

### TC-S-02 · Obstructed direct path yields no direct shot

**Scenario**: A third ball sits between the cue ball and an object ball, blocking the direct path.

**Preconditions**: Three balls arranged so one is squarely between cue and object (perpendicular distance < 57.15 mm).

**Steps**:
1. Upload the photo with the blocking ball in place.
2. Inspect `direct_shots`.

**Expected**:
- No `direct` shot is returned for that `(cue, object)` pair.
- The blocking ball's position causes the path to be discarded.

---

### TC-S-03 · Direct shot to all six pocket types

**Scenario**: Verify that any of the 6 pockets can be a valid direct shot target.

**Steps**:
1. For each pocket (`TL`, `TR`, `ML`, `MR`, `BL`, `BR`), upload a photo where a direct shot to that pocket exists.
2. Confirm `pocket_id` matches in `direct_shots`.

**Expected**:
- Each of the 6 pockets appears as `pocket_id` in at least one `direct_shots` result across the 6 test photos.

---

### TC-S-04 · Direct shots appear before bank shots in results UI

**Scenario**: The UI shows "Direct Shots" group above "Bank Shots" group.

**Steps**:
1. Upload a photo where both direct and bank shots exist.
2. Observe the result panel.

**Expected**:
- "Direct Shots" section label appears first.
- "Bank Shots" section label appears below it.
- Shot items within each group are present.

---

## TC-B — Bank Shot Detection

### TC-B-01 · One-bank shot detected off RIGHT rail into ML pocket

**Scenario**: Replicates the worked example from SPEC §2.1.

**Preconditions**: Photo with cue ball on the left side, 8-ball near the centre-right, no obstructions.

**Steps**:
1. Upload photo.
2. Inspect `bank_shots` in response.

**Expected**:
- At least one entry has `shot_type: "one_bank"`.
- `rail: "RIGHT"`.
- `pocket_id: "ML"`.
- `contact_point.x` ≈ `2540` (right rail x-coordinate) ± tolerance.
- `path` has 4 waypoints: `[cue, object, rail_contact, pocket]`.
- `bank_angle_deg` is a positive number < 90.

---

### TC-B-02 · One-bank shot detected off each of the four rails

**Scenario**: The system can find bank shots off all four rails (TOP, BOTTOM, LEFT, RIGHT).

**Steps**:
1. For each rail, upload a table photo where a valid bank off that rail exists.
2. Inspect `bank_shots[].rail`.

**Expected**:
- Across 4 test uploads, each rail value (`TOP`, `BOTTOM`, `LEFT`, `RIGHT`) appears as a valid bank shot rail at least once.

---

### TC-B-03 · Contact point inside a pocket opening is rejected

**Scenario**: A geometric bank path whose contact point lands inside the pocket opening (not on valid cushion) is discarded.

**Preconditions**: Photo with geometry that would produce a contact point at the side pocket opening on a long rail.

**Steps**:
1. Upload the photo.
2. Inspect `bank_shots`.

**Expected**:
- No bank shot is returned with a contact point whose coordinates fall within the pocket exclusion zone (within `side_pocket_radius ≈ 63 mm` of a side pocket centre on the rail edge).

---

### TC-B-04 · Post-rail path blocked by third ball — bank shot rejected

**Scenario**: A valid geometric bank is blocked because a ball sits between the rail contact point and the pocket.

**Preconditions**: A third ball is placed on the reflected path between the rail contact point and the target pocket.

**Steps**:
1. Upload photo with the blocking ball.
2. Check `bank_shots`.

**Expected**:
- No bank shot is returned for that `(object ball, rail, pocket)` combination.

---

### TC-B-05 · Cross-table bank (object ball near left → RIGHT rail → ML pocket)

**Scenario**: Object ball near the left side banks off the right rail and returns to the left side pocket.

**Steps**:
1. Upload appropriate photo.
2. Inspect `bank_shots`.

**Expected**:
- Shot returned with `rail: "RIGHT"` and `pocket_id: "ML"`.
- `contact_point.x` is close to `2540`.
- `path[3]` (pocket waypoint) is close to `{x: 0, y: 635}` (ML coordinates).

---

### TC-B-06 · Bank shots sorted by ease\_score ascending

**Scenario**: Multiple valid bank shots are returned sorted by `ease_score` (lowest first).

**Steps**:
1. Upload a photo with multiple valid bank shots of different angles.
2. Inspect `bank_shots` ordering.

**Expected**:
- `bank_shots[0].ease_score ≤ bank_shots[1].ease_score ≤ ...`
- `ease_score = |bank_angle_deg - 90|` for each shot.
- The shot with `bank_angle_deg` closest to 90° appears first.

---

### TC-B-07 · 90° bank has ease\_score = 0

**Scenario**: A ball that travels exactly perpendicular to a rail has the lowest possible ease score.

**Preconditions**: Controlled geometry where the object ball departs at exactly 90° to a rail.

**Steps**:
1. Upload photo or use synthetic API test input.
2. Check `bank_shots[0].ease_score`.

**Expected**:
- `ease_score` equals `0` (or < 1 with floating-point tolerance).
- `bank_angle_deg` equals `90` (± 1°).

---

### TC-B-08 · Bank shot has all required fields

**Scenario**: Every field in the `BankShot` schema is present and correctly typed.

**Steps**:
1. Upload a photo that produces at least one bank shot.
2. Inspect the first element of `bank_shots`.

**Expected**:
- `shot_type: "one_bank"` ✓
- `cue_ball: { x, y }` — both numbers ✓
- `object_ball_id` — non-empty string ✓
- `object_ball_label` — matches `"solid-<hue>"` | `"stripe-<hue>"` | `"eight"` ✓
- `rail` — one of `"TOP"`, `"BOTTOM"`, `"LEFT"`, `"RIGHT"` ✓
- `contact_point: { x, y }` — both numbers within table bounds ✓
- `path` — array of exactly **4** `{ x, y }` objects ✓
- `bank_angle_deg` — number in range `(0, 180)` ✓
- `ease_score` — non-negative number ✓
- `pocket_id` — one of the 6 valid IDs ✓

---

## TC-R — Results UI

### TC-R-01 · Result screen: annotated image displayed full-width at top

**Scenario**: The result layout is mobile-first (image on top, list below).

**Steps**:
1. Upload a valid photo from a mobile browser (or narrow viewport ≤ 430 px wide).
2. Observe the Result Screen layout.

**Expected**:
- The annotated image occupies the **full width** of the viewport.
- The shot list is below the image, not beside it.
- No horizontal scrolling is required.

---

### TC-R-02 · Result screen: "Direct Shots" group label visible

**Steps**:
1. Upload a photo with at least one direct shot.
2. Observe the Result Screen.

**Expected**:
- A clearly visible **"Direct Shots"** section heading is present.
- Shot items under it show format: `"[color/type] ball directly into [pocket]"`.

---

### TC-R-03 · Result screen: "Bank Shots" group label visible

**Steps**:
1. Upload a photo with at least one bank shot.
2. Observe the Result Screen.

**Expected**:
- A clearly visible **"Bank Shots"** section heading is present below the Direct Shots section.
- Shot items show format: `"[color/type] ball via [rail] rail into [pocket]"`.

---

### TC-R-04 · Tapping a shot highlights its path on the image

**Steps**:
1. Upload a photo with multiple valid shots.
2. On the Result Screen, tap the second shot in the Bank Shots list.
3. Observe the annotated image.

**Expected**:
- The selected shot's path is highlighted on the image:
  - 🔵 Blue arrow: cue → object ball.
  - 🟠 Orange dashed arrow: object ball → rail contact.
  - 🟢 Green dashed arrow: rail contact → pocket.
  - 🟡 Yellow filled circle on target ball.
  - 🟣 Purple pulsing ring on target pocket.
- Previously highlighted shot path is de-highlighted.

---

### TC-R-05 · First shot is highlighted by default

**Steps**:
1. Upload a valid photo with results.
2. Observe the Result Screen without tapping anything.

**Expected**:
- The first shot in the ranked list (lowest `ease_score`) is **automatically highlighted** on the annotated image.

---

### TC-R-06 · Ball label shown without number in shot list

**Steps**:
1. Upload a photo with numbered balls visible.
2. Read the shot list text.

**Expected**:
- Shot descriptions use color/type label only (e.g. `"solid-red ball via RIGHT rail into ML"`).
- No ball number (e.g. `"3-ball"` or `"Ball 3"`) appears anywhere in the shot list.

---

### TC-R-07 · Annotated image — cue ball labelled "CUE"

**Steps**:
1. Upload a valid photo and observe the annotated image.

**Expected**:
- The detected cue ball has a white circle outline and the text label `"CUE"` on the image.

---

## TC-E — Error & Edge Cases

### TC-E-01 · Non-pool-table image shows error

**Scenario**: User uploads a photo that does not contain a pool table.

**Steps**:
1. Upload a photo of a landscape, indoor room, or any non-pool scene.
2. Observe the response.

**Expected**:
- Error Screen is shown.
- Message displayed: _"No valid shots found — every possible bank is blocked or misses all pockets."_ or a table-not-detected variant.
- No `balls`, `direct_shots`, or `bank_shots` data is displayed.

---

### TC-E-02 · Photo with no cue ball shows error

**Scenario**: Table photo has object balls but no white cue ball.

**Steps**:
1. Upload a photo where the cue ball is absent (e.g. it has been pocketed).
2. Observe response.

**Expected**:
- Error Screen is shown.
- No shots are listed (shots require a cue ball to compute paths from).

---

### TC-E-03 · All bank paths blocked — no shots found

**Scenario**: Table position where every object ball is surrounded by blockers and no valid bank geometry leads to a pocket.

**Steps**:
1. Upload a photo (or synthetic test) with all paths obstructed.
2. Observe Result Screen.

**Expected**:
- `direct_shots: []` and `bank_shots: []` in the API response.
- Error Screen shown: `"No valid shots found — every possible bank is blocked or misses all pockets."`

---

### TC-E-04 · Partially visible table is rejected

**Scenario**: The photo cuts off before the full table is visible.

**Steps**:
1. Upload a cropped photo showing only half the table.
2. Observe response.

**Expected**:
- Error screen shown (table boundary detection fails or returns partial result).
- No shot analysis is performed on incomplete geometry.

---

### TC-E-05 · Overlapping balls — no crash

**Scenario**: Two balls are touching or nearly overlapping in the photo.

**Steps**:
1. Upload a photo with two balls touching each other.
2. Inspect `balls` array.

**Expected**:
- The system detects at least one of the two balls (the more distinct one).
- No crash or server error (`5xx`) occurs.
- Response is valid JSON conforming to the `AnalysisResult` schema.

---

### TC-E-06 · Oversized file (> 20 MB) is rejected client-side

**Steps**:
1. Attempt to upload a 25 MB image file.

**Expected**:
- File is rejected **before** any HTTP request to the server.
- Error shown: file size exceeds 20 MB.
- `POST /analyze` is **not** called (verify in browser network tab).

---

### TC-E-07 · API response schema validation

**Scenario**: Verify the backend always returns a valid `AnalysisResult` schema.

**Steps**:
1. Upload a valid photo.
2. Validate the JSON response against the `AnalysisResult` TypeScript interface.

**Expected**:
- `table_dims_mm`: `{ width: number, height: number }` ✓
- `pockets`: array of 6 objects, each with `id`, `x`, `y`, `radius_mm` ✓
- `balls`: array of objects each with `id`, `label`, `x`, `y`, `radius_mm` ✓
- `direct_shots`: array (may be empty) ✓
- `bank_shots`: array (may be empty) ✓
- No extra unexpected top-level fields present.

---

### TC-E-08 · HTTP 415 on wrong Content-Type

**Scenario**: API called with wrong content type.

**Steps**:
1. Send `POST /analyze` with `Content-Type: application/json` (instead of `multipart/form-data`).

**Expected**:
- Server responds with `HTTP 415 Unsupported Media Type` or `HTTP 422 Unprocessable Entity`.
- Response body contains a descriptive error message.

---

## Test Coverage Matrix

| Spec Requirement | Test Cases |
|---|---|
| Upload via file picker | TC-U-01 |
| Upload via drag-and-drop | TC-U-02, TC-U-03 |
| Format validation (JPG/PNG/WEBP only) | TC-U-04 |
| File size limit (20 MB) | TC-U-05, TC-E-06 |
| Processing spinner | TC-U-06 |
| Automatic table boundary detection | TC-D-01 |
| 6 pocket identification | TC-D-01 |
| Cue ball detection & classification | TC-D-02 |
| Solid ball label | TC-D-03 |
| Stripe ball label | TC-D-04 |
| 8-ball label | TC-D-05 |
| No number OCR | TC-D-06 |
| Coordinate mapping (mm, not pixels) | TC-D-07 |
| Direct shot detection | TC-S-01 |
| Obstruction check (direct) | TC-S-02 |
| All 6 pockets as direct shot targets | TC-S-03 |
| Direct Shots shown before Bank Shots | TC-S-04 |
| Bank shot off RIGHT rail → ML | TC-B-01 |
| All four rails tested | TC-B-02 |
| Pocket exclusion zone rejects contact | TC-B-03 |
| Post-rail obstruction check | TC-B-04 |
| Cross-table bank geometry | TC-B-05 |
| ease_score sorting ascending | TC-B-06 |
| 90° bank ease_score = 0 | TC-B-07 |
| BankShot schema completeness | TC-B-08 |
| Mobile-first layout (image top) | TC-R-01 |
| "Direct Shots" group label | TC-R-02 |
| "Bank Shots" group label | TC-R-03 |
| Tap-to-highlight shot path | TC-R-04 |
| First shot auto-highlighted | TC-R-05 |
| No ball numbers in UI | TC-R-06 |
| Cue ball "CUE" label on image | TC-R-07 |
| Non-pool image error | TC-E-01 |
| No cue ball error | TC-E-02 |
| All paths blocked → no shots error | TC-E-03 |
| Partial table rejected | TC-E-04 |
| Overlapping balls handled gracefully | TC-E-05 |
| API schema validation | TC-E-07 |
| Wrong Content-Type → 415 | TC-E-08 |

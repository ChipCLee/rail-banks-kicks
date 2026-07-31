# Rail-Kick: Pool Table Shot Detection System

## Overview

**Rail-Kick** is a browser-based web application that analyzes a pool/billiards table photograph to:

1. Detect and locate all balls on the table.
2. Identify any **one-bank shot** where the cue ball hits an object ball directly and the object ball bounces off one rail into a pocket.
3. Present the result as an annotated image highlighting the **target ball**, the **rail contact point**, and the **target pocket** for each valid shot found.

> **v1 scope**: One-bank shots only. Kick shots (cue ball off rail first) and multi-rail combinations are deferred to v2.

---

## Table of Contents

- [Glossary](#glossary)
- [System Architecture](#system-architecture)
- [Web Application](#web-application)
- [Feature 1 – Ball Position Analysis](#feature-1--ball-position-analysis)
- [Feature 2 – One-Bank Shot Detection](#feature-2--one-bank-shot-detection)
- [Data Models](#data-models)
- [Constraints & Assumptions](#constraints--assumptions)
- [v2 Scope (Planned)](#v2-scope-planned)
  - [Feature 3 – One-Rail Kick Shot Detection](#feature-3--one-rail-kick-shot-detection)
  - [Feature 4 – Cushion Throw Modelling](#feature-4--cushion-throw-modelling)
- [Open Questions](#open-questions)

---

## Glossary

| Term | Definition |
|---|---|
| **Cue ball** | The white ball struck directly by the cue stick. |
| **Object ball** | Any non-white ball on the table. |
| **Rail** | The cushioned inner edge of the table boundary. There are four rails: Top, Bottom, Left, Right. |
| **Bank shot** | A shot where the object ball contacts a rail before going into a pocket. |
| **One-rail kick** | Cue ball travels in a straight line, bounces off exactly one rail, then travels in a straight line to hit the object ball. |
| **One-rail bank** | Object ball travels in a straight line, bounces off exactly one rail, then travels in a straight line into a pocket. |
| **Pocket** | One of six scoring holes: four corner pockets and two side (middle) pockets. |
| **Diamond system** | A geometric aid using reference points (diamonds) marked on the rails for calculating rail-kick angles. |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Web App (Browser)                           │
│  • Photo upload UI (drag-and-drop or file picker)               │
│  • Shot result display (annotated image overlay)                │
│  • Shot list panel (target ball + pocket per valid shot)        │
└──────────────────────────┬──────────────────────────────────────┘
                           │  image bytes (HTTP POST)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Backend API (REST)                          │
│  POST /analyze  →  returns AnalysisResult JSON                  │
└──────────┬───────────────────────────────────────────┬──────────┘
           │                                           │
           ▼                                           ▼
┌──────────────────────────┐             ┌─────────────────────────┐
│  Computer Vision Module  │             │  Geometry/Physics Module │
│  • Table boundary detect │             │  • Ghost-ball direct-hit │
│  • Ball detection        │             │    path check            │
│  • Ball classification   │             │  • Object ball → rail    │
│    (color/type only,     │             │    reflection calc       │
│     no number OCR)       │             │  • Pocket intersection   │
│  • Coordinate mapping    │             │    test (one-bank only)  │
└──────────────────────────┘             └─────────────────────────┘
```

---

## Web Application

### Stack

| Layer | Technology |
|---|---|
| Frontend | React (Vite) + Vanilla CSS |
| Backend API | Python (FastAPI) or Node.js (Express) |
| CV processing | OpenCV.js (client-side) **or** Python OpenCV (server-side) |
| Deployment | Single-page app + lightweight REST API; deployable via Docker |

> **Preferred approach**: Run OpenCV on the **backend** (Python + FastAPI) to avoid the 8 MB OpenCV.js bundle. The frontend is a pure React SPA that uploads a photo and renders the response.

### User Flow

```
1. User opens the web app in a browser.
2. User uploads a photo of the pool table by holding phone overhead
   (drag-and-drop area or "Choose File" button).
3. App sends the photo to POST /analyze.
4. Backend (Python + FastAPI + OpenCV) processes and returns AnalysisResult JSON.
5. App renders the annotated image full-width at the top.
6. App shows a scrollable shot list below, in two groups:
   - "Direct Shots" → "[color/type] ball directly into [pocket]" (ranked easiest first)
   - "Bank Shots"   → "[color/type] ball via [rail] rail into [pocket]" (ranked easiest first)
7. User taps a shot in the list to highlight its specific path on the image.
```

### UI Screens

| Screen | Description |
|---|---|
| **Upload Screen** | Full-page drop zone with instructions. Accepts JPG/PNG/WEBP ≤ 20 MB. |
| **Processing Screen** | Spinner / progress indicator while backend analyzes. |
| **Result Screen** | **Mobile-first layout**: annotated image full-width at top; shot list scrolls below. Two labelled groups: "Direct Shots" (ranked by ease) then "Bank Shots" (ranked by ease). Tapping a shot highlights its path on the image. |
| **Error Screen** | Friendly message: `"No valid shots found — every possible bank is blocked or misses all pockets."` Also shown if no table or cue ball is detected in the photo. |

---

## Feature 1 – Ball Position Analysis

### 1.1 Input

The user uploads a single photo through the web UI.

| Field | Type | Description |
|---|---|---|
| `image` | `multipart/form-data file` | A JPG, PNG, or WEBP photograph of the pool table. Taken by **holding a phone overhead** (moderate angle, common case). Maximum file size: 20 MB. |

> **No camera metadata is required.** The perspective correction is computed automatically from the detected table boundary using **full automatic CV detection** (green felt masking + corner finding). There is no manual calibration step in v1.

### 1.2 Processing Steps

#### Step 1 – Table Boundary Detection

1. Detect the **green felt region** using HSV color masking.
2. Find the **four corner pockets** as the corners of the felt region (Hough line method or contour-based).
3. Apply a **perspective homography transform** to produce a canonical top-down view of the table with known real-world dimensions (9-foot table: 254 cm × 127 cm playfield, or 8-foot: 224 cm × 112 cm).
4. Mark all **six pocket centres** in table coordinates:
   - Four corners (TL, TR, BL, BR)
   - Two side centres (ML, MR)

#### Step 2 – Ball Detection

1. Apply **Gaussian blur** to reduce noise.
2. Use **Hough Circle Transform** (`cv2.HoughCircles`) to find circular regions.
3. Filter circles by:
   - Radius range matching a standard pool ball diameter (57.15 mm ≈ `r_pixels` derived from table scale).
   - Overlap removal (non-maximum suppression on confidence).

#### Step 3 – Ball Classification

Classify each detected ball by colour profile of the circular ROI. **Ball number recognition (OCR) is explicitly out of scope.**

| Ball Type | Colour Signature | Label Used in Output |
|---|---|---|
| Cue ball | High brightness, near-white HSV | `"cue"` |
| Solid balls | Dominant single hue covering most of ROI | `"solid-<hue>"` e.g. `"solid-red"` |
| Stripe balls | White body with a dominant hue band | `"stripe-<hue>"` e.g. `"stripe-blue"` |
| 8-ball | Dark / near-black dominant | `"eight"` |

Hue labels are derived from the dominant HSV hue bucket (red, orange, yellow, green, blue, purple, maroon). No number is assigned.

Return the result as a **BallMap**: a list of `Ball` objects (see [Data Models](#data-models)).

#### Step 4 – Coordinate Output

Each ball's pixel centre is transformed via the homography matrix into **table coordinates** `(x, y)` expressed in millimetres from the bottom-left corner of the playfield.

### 1.3 Output

```json
{
  "table_dims_mm": { "width": 2540, "height": 1270 },
  "pockets": [
    { "id": "TL", "x": 0,    "y": 1270 },
    { "id": "TR", "x": 2540, "y": 1270 },
    { "id": "ML", "x": 0,    "y": 635  },
    { "id": "MR", "x": 2540, "y": 635  },
    { "id": "BL", "x": 0,    "y": 0    },
    { "id": "BR", "x": 2540, "y": 0    }
  ],
  "balls": [
    { "id": "cue",  "label": "cue",         "x": 635,  "y": 635, "radius_mm": 28.6 },
    { "id": "obj1", "label": "solid-red",   "x": 1270, "y": 900, "radius_mm": 28.6 },
    { "id": "obj2", "label": "stripe-blue", "x": 800,  "y": 400, "radius_mm": 28.6 }
  ]
}
```

---

## Feature 2 – One-Bank Shot Detection

> **v1 scope**: This feature detects **two shot types** for every object ball on the table:
> 1. **Direct shots** — cue ball hits object ball, object ball goes straight to a pocket (no rail).
> 2. **One-bank shots** — cue ball hits object ball directly, object ball bounces off exactly one rail into a pocket.
>
> All 6 pockets (4 corners + 2 sides) are valid targets. Kick shots (cue ball off rail first) are deferred to v2.
> A path is **blocked** if any other ball's body comes within one ball-diameter (57.15 mm) of the line segment (ghost-ball clearance check).

### 2.1 Concept & Worked Example

**Scenario**: The 8-ball sits near the center of the table. The cue ball is on the left side. There is no clear straight path to any pocket, but the 8-ball can be banked off the right rail into the left side pocket (ML).

```
  ML●                              ●MR
  ┌────────────────────────────────┐
  │                                │
  │  C●                    ●8     │   ← object ball near right rail
  │    ╲                  ╱       │
  │     ╲ cue→object     ╱        │
  │      ●────────────►●          │
  │                     ╲         │
  │             bank path ╲       │
  │                        ►──●──►│  ← bounces off RIGHT rail
  │                        contact│
  │        ◄───────────────┘      │
  │  target pocket: ML ●          │
  └────────────────────────────────┘
```

The object ball hits the **right rail** and travels to the **left side pocket (ML)** — the pocket on the **other side** of the table from the rail it hit.

### 2.2 Algorithm

For each `(cue_ball, object_ball)` pair, test all four rails as candidates.

#### Step 1 – Direct Hit Check (Cue Ball → Object Ball)

Verify the cue ball can reach the object ball in a **straight line without obstruction**.

- Compute the line segment from `C` (cue ball centre) to `O` (object ball centre).
- For every other ball `B`, check: `perp_distance(B.centre, segment C→O) < ball_diameter` (57.15 mm).
- If any ball blocks the path → this object ball is **not reachable**; skip all rails for it.

#### Step 2 – Ghost-Ball Contact Point

The cue ball does not hit the exact centre of the object ball. Find the **ghost-ball position** `G` — the point where the cue ball's centre would be when it just touches the object ball:

```
# Direction from C toward O
d = normalize(O - C)
# Ghost ball centre is one ball-diameter back along that direction from O
G = O - d * ball_diameter   # = O - d * 57.15mm
```

The object ball departs from `O` along the direction `d` after being struck.

#### Step 3 – Rail Reflection (Object Ball Path)

For each candidate rail, reflect the object ball's **departure direction** off that rail:

```
# Example: object ball travels in direction d = (dx, dy)
# Reflecting off LEFT rail (x = 0): flip the x component
d_reflected = (-dx, dy)

# Find the intersection point P of the object ball's path with the rail
# From O in direction d, find where it hits x = 0 (LEFT rail)
t = (0 - ox) / dx          # solve O.x + t*dx = 0
P = (0, oy + t * dy)       # contact point on LEFT rail
```

Apply symmetrically for all four rails.

#### Step 4 – Rail Contact Point Validity

The contact point `P` must fall within the **active cushion range** (not inside a pocket opening):

| Rail | Valid range |
|---|---|
| LEFT / RIGHT | `y ∈ [corner_pocket_radius, table_height - corner_pocket_radius]`, excluding the side pocket zone `y ∈ [table_height/2 - side_pocket_radius, table_height/2 + side_pocket_radius]` |
| TOP / BOTTOM | `x ∈ [corner_pocket_radius, table_width - corner_pocket_radius]`, excluding the side pocket zone `x ∈ [table_width/2 - side_pocket_radius, table_width/2 + side_pocket_radius]` |

Typical values: `corner_pocket_radius ≈ 57 mm`, `side_pocket_radius ≈ 63 mm`.

#### Step 5 – Pocket Intersection Check

From `P`, the object ball travels in the reflected direction `d_reflected`. Check whether this path passes within `pocket_radius` of any of the 6 pocket centres.

- If yes → **valid bank shot found**; record the pocket as the `pocket_id`.
- If no → discard this rail candidate.

#### Step 6 – Post-Rail Obstruction Check

Verify the reflected path from `P` to the target pocket is not blocked by any other ball (same perpendicular-distance test as Step 1).

### 2.3 Shot Output

Each valid shot is emitted as a `Shot` record. Every one-bank shot always has both an `object_ball_id` (which ball to strike) and a `pocket_id` (where it will fall).

**Example – 8-ball banks off RIGHT rail into ML pocket:**
```json
{
  "shot_type": "one_bank",
  "cue_ball": { "x": 400, "y": 635 },
  "object_ball_id": "obj_eight",
  "object_ball_label": "eight",
  "rail": "RIGHT",
  "contact_point": { "x": 2540, "y": 480 },
  "path": [
    { "x": 400,  "y": 635 },
    { "x": 1800, "y": 635 },
    { "x": 2540, "y": 480 },
    { "x": 0,   "y": 635 }
  ],
  "bank_angle_deg": 38.0,
  "pocket_id": "ML"
}
```

> `path` waypoints: `[cue_ball_centre, object_ball_centre, rail_contact_point, pocket_centre]`

#### Shot Ranking (Q5)

The `shots` array in `AnalysisResult` is **sorted by ease of shot**, easiest first.

**Ease metric** = how close the bank angle is to **90°** (a ball hitting the rail perfectly square is the easiest to judge and execute accurately). Shallower angles are harder because small aiming errors cause large positional errors after the rebound.

```
ease_score = |bank_angle_deg - 90|   // lower = easier
```

| `bank_angle_deg` | `ease_score` | Difficulty |
|---|---|---|
| 90° | 0 | Easiest — ball hits rail square |
| 60° or 120° | 30 | Moderate |
| 30° or 150° | 60 | Hard — very shallow or steep |

The result panel displays shots in this order, with the #1 shot (lowest `ease_score`) highlighted by default.

### 2.4 Annotated Image Output

The backend returns the perspective-corrected top-down image with overlays. The frontend renders this and allows the user to tap a shot in the results panel to highlight its specific path.

| Element | Visual Style | Meaning |
|---|---|---|
| Cue ball | ⚪ White circle outline + label "CUE" | Identified cue ball |
| Object balls | Colour-matched circle outline + `label` text | Each object ball — no number shown |
| **Target ball** (active shot) | 🟡 Bright yellow filled circle | Object ball to be struck |
| Cue → object ball | 🔵 Blue solid arrow | Direct hit path (cue ball to object ball) |
| Object ball → rail contact | 🟠 Orange dashed arrow | Object ball travelling to the rail |
| Rail contact → pocket | 🟢 Green dashed arrow | Object ball path after banking off rail |
| Rail contact point | ⬜ White filled dot | Where the object ball hits the cushion |
| **Target pocket** (active shot) | 🟣 Purple pulsing ring | Pocket the banked ball enters |

---

## Data Models

```typescript
// Ball number recognition is NOT included.
// Balls are identified by color/type label only.
interface Ball {
  id: string;            // "cue" | "obj1" | "obj2" | ...
  label: string;         // "cue" | "solid-<hue>" | "stripe-<hue>" | "eight"
                         // e.g. "solid-red", "stripe-blue"
  x: number;             // mm from bottom-left of playfield
  y: number;             // mm from bottom-left of playfield
  radius_mm: number;     // standard: 28.575
}

interface Pocket {
  id: "TL" | "TR" | "ML" | "MR" | "BL" | "BR";
  x: number;             // mm from bottom-left
  y: number;             // mm from bottom-left
  radius_mm: number;     // corner ≈ 57mm, side ≈ 63mm
}

type RailId = "TOP" | "BOTTOM" | "LEFT" | "RIGHT";

// v1 supports two shot types. Kick shots ('one_rail_kick') are v2.
interface DirectShot {
  shot_type: "direct";
  cue_ball: { x: number; y: number };
  object_ball_id: string;       // key into balls[]
  object_ball_label: string;    // e.g. "eight", "solid-red"
  path: Array<{ x: number; y: number }>;    // [cue, object, pocket]
  ease_score: number;           // |0 - 0| = 0 for direct (always easiest)
  pocket_id: string;
}

interface BankShot {
  shot_type: "one_bank";
  cue_ball: { x: number; y: number };
  object_ball_id: string;
  object_ball_label: string;
  rail: RailId;                 // which rail the object ball banks off
  contact_point: { x: number; y: number };  // where object ball hits the rail
  path: Array<{ x: number; y: number }>;    // [cue, object, rail_contact, pocket]
  bank_angle_deg: number;       // object ball's angle of incidence at the rail
  ease_score: number;           // |bank_angle_deg - 90| — lower is easier
  pocket_id: string;
}

type Shot = DirectShot | BankShot;

interface AnalysisResult {
  table_dims_mm: { width: number; height: number };
  pockets: Pocket[];
  balls: Ball[];
  direct_shots: DirectShot[];   // sorted by ease_score ascending
  bank_shots: BankShot[];       // sorted by ease_score ascending
}
```

---

## Constraints & Assumptions

| # | Constraint |
|---|---|
| C1 | Table must be a standard 7-foot, 8-foot, or 9-foot pool table. Snooker tables are out of scope for v1. |
| C2 | The photograph must capture the **entire** table surface. Partial views are not supported. |
| C3 | Accepted image formats: **JPG, PNG, WEBP**. Maximum file size: **20 MB**. |
| C4 | Lighting must be reasonably uniform. Extreme shadows that mask balls are not handled. |
| C5 | v1 detects **two shot types**: (a) direct shots — cue ball hits object ball straight to a pocket; (b) one-bank shots — object ball hits exactly one rail before pocketing. The cue ball must reach the object ball directly in both cases. |
| C6 | Multi-rail banks (2+ rails) are out of scope for v1. |
| C7 | **Kick shots** (cue ball off a rail before hitting the object ball) are out of scope for v1. |
| C8 | Ball cushion compression, throw, squirt, and spin are **not modelled** in v1; pure geometric reflection is used. |
| C9 | **Ball number recognition is out of scope.** Balls are identified by color/type only. |
| C10 | The cue ball must be visually distinguishable (white or near-white). |
| C11 | A minimum of 2 balls (cue ball + at least one object ball) must be present on the table. |
| C12 | Balls must not be completely overlapping in the image. |
| C13 | The application runs in a **web browser**. No native app installation is required. |

---

## v2 Scope (Planned)

> Features below are **not part of v1**. They are documented here to guide future architecture decisions and to ensure the v1 data model remains extensible.

---

### Feature 3 – One-Rail Kick Shot Detection

A **kick shot** is the mirror image of a bank shot: the **cue ball** bounces off a rail first, then travels to strike the object ball. The object ball then pockets by any means (direct or bank).

#### 3.1 Concept & Worked Example

**Scenario**: The 8-ball is blocked from a direct hit. The player aims the cue ball at the **front rail at the 2.5-diamond mark**, the cue ball bounces back and strikes the 8-ball, which then rolls into the **lower-left corner pocket (BL)**.

```
BL●                              ●BR   ← bottom (foot) rail
  ┌────────────────────────────────┐
  │    ◄────────────────●◄────── │  ← 2.5-diamond contact on front rail
  │  ►                  ╲         │
  │  ● C (cue ball)       ○ 8-ball │
  │  kick path ►          ╲        │
  │                        ►       │
  │              8-ball → BL pocket│
  ┌────────────────────────────────┐
  ●────────────────────────────────●   ← front (head) rail
TL                              TR
```

#### 3.2 Diamond Coordinate System

Pool tables have evenly spaced **diamond markers** on the rail cushions used as aiming references.

| Table size | Diamonds per long rail | Diamonds per short rail |
|---|---|---|
| 9-foot | 7 | 3 |
| 8-foot | 7 | 3 |
| 7-foot | 7 | 3 |

Diamonds are numbered from **0 (corner pocket) to 4 (side pocket)** along each half-rail.

```
Corner          Side          Corner
  ●───◆───◆───◆───◆●◆────◆───◆───◆───◆───●
  0   1   2   3   4  4   3   2   1   0
```

The **2.5-diamond mark** is the midpoint between diamonds 2 and 3 on the short (head/foot) rail:

```
rail_contact_x = corner_x + 2.5 * (rail_length / 4)
```

The system expresses every rail contact point in diamond units for display (e.g. `"2.5 diamonds from TL corner on TOP rail"`).

#### 3.3 Algorithm

##### Step 1 – Enumerate Rail Contact Candidates

For each `(cue_ball, object_ball)` pair and each of the four rails, compute the cue ball contact point using the **mirror / reflection method** (same as v1 bank, but mirrored on the cue ball side):

```
# Mirror the OBJECT BALL across the chosen rail
O_mirror = reflect(O, rail)

# Draw a straight line from CUE BALL to the mirrored position
# The intersection with the rail is the kick contact point P
t = (rail_coord - cue.y) / (O_mirror.y - cue.y)   # example: top rail
P = (cue.x + t * (O_mirror.x - cue.x), rail_coord)
```

##### Step 2 – Diamond Label

Convert the contact point `P` from mm to diamond units for display:

```
diamond_position = (P.x - corner_x) / diamond_spacing_mm
```

##### Step 3 – Obstruction Checks

1. **Cue ball → `P`**: path must be clear (same perpendicular-distance test as v1).
2. **`P` → Object ball**: path after rail bounce must be clear.

##### Step 4 – Object Ball Pocketability

After the cue ball strikes the object ball, check if the object ball can be pocketed (direct or one-bank). In v2 this is limited to **direct pocket** (object ball → pocket in a straight line) to keep scope manageable.

#### 3.4 Shot Output (v2 extension)

Kick shots extend the `Shot` interface with `shot_type: "one_rail_kick"` and an additional `diamond_label` field:

```json
{
  "shot_type": "one_rail_kick",
  "cue_ball": { "x": 400, "y": 900 },
  "object_ball_id": "obj_eight",
  "object_ball_label": "eight",
  "rail": "TOP",
  "contact_point": { "x": 1270, "y": 1270 },
  "diamond_label": "2.5 diamonds from TL on TOP rail",
  "path": [
    { "x": 400,  "y": 900  },
    { "x": 1270, "y": 1270 },
    { "x": 1800, "y": 635  },
    { "x": 0,   "y": 0    }
  ],
  "bank_angle_deg": 45.0,
  "pocket_id": "BL"
}
```

> `path` waypoints: `[cue_ball, rail_contact_point, object_ball, pocket]`

#### 3.5 Annotated Image (additions for kick shots)

| Element | Visual Style | Meaning |
|---|---|---|
| Cue → rail contact | 🔵 Blue solid arrow | Cue ball travelling to the rail |
| Rail contact → object ball | 🟠 Orange solid arrow | Cue ball after bouncing off rail |
| Object ball → pocket | 🟢 Green dashed arrow | Object ball's path to pocket |
| Diamond marker | 🔷 Blue diamond icon on rail | Rail contact point expressed in diamonds |

---

### Feature 4 – Cushion Throw Modelling

#### 4.1 What is Cushion Throw?

In real play, when a ball strikes a cushion at a shallow angle, **friction between the ball and the rubber** causes the rebound angle to be slightly larger than the geometric angle (the ball is "grabbed" by the cushion and steered back toward the rail it came from). This effect is called **cushion throw** or **rail throw**.

```
  Geometric reflection:       Actual rebound (throw applied):

  ───────►╲                  ────────────►╲
            ╲ θ_in              θ_in         ╲ θ_out < θ_in
             ╲────►                            ╲────►
```

#### 4.2 Throw Correction Model (v2)

v2 applies a simplified empirical **throw correction** based on angle of incidence:

```
# Throw is most significant at shallow angles, negligible near 90°
throw_correction_deg = k * cos(2 * bank_angle_deg)

# Adjusted rebound angle
rebound_angle_deg = bank_angle_deg - throw_correction_deg
```

Where `k` is an empirically tuned constant (typical value: **3–6 degrees** maximum throw). This is applied to both bank shots (v1 upgraded) and kick shots (v2).

| Angle of incidence | Throw correction (approx.) |
|---|---|
| 90° (square) | 0° (no throw) |
| 60° | ~1.5° |
| 45° | ~3° |
| 30° | ~4.5° |
| 15° (very shallow) | ~5.5° |

#### 4.3 Impact on v1 One-Bank Shots

In v2, the throw correction is also **retroactively applied to one-bank shots** from v1. The pocket intersection check and shot validity are re-evaluated using the corrected rebound angle. This may change which shots are listed as valid for near-miss angles.

#### 4.4 Shot Output (v2 extension)

The `Shot` record gains a `throw_correction_deg` field:

```json
{
  "shot_type": "one_bank",
  "bank_angle_deg": 38.0,
  "throw_correction_deg": 4.1,
  "adjusted_rebound_angle_deg": 33.9,
  "pocket_id": "ML"
}
```

---

## Open Questions

| # | Status | Question | Impact |
|---|---|---|---|
| Q1 | ✅ Resolved | Should **cushion throw** be modelled? **Yes** — planned for v2. v1 uses pure geometric reflection; v2 will apply a throw offset. | — |
| Q2 | ✅ Resolved | Input is a **photo uploaded via the web UI** (drag-and-drop or file picker). Live video is out of scope for v1. | — |
| Q3 | ✅ Resolved | Snooker table support: **No**, not planned. Pool tables only (7 / 8 / 9 ft). | — |
| Q4 | ✅ Resolved | **Number recognition is not required.** Color/type label is sufficient. | — |
| Q5 | ✅ Resolved | **Yes — shots are ranked by ease**, easiest first. See ranking definition in §2.3. | — |
| Q6 | ✅ Resolved | Target platform is **web browser** (React SPA + Python backend). Stack selected: Vite + FastAPI + OpenCV. | — |
| Q7 | ✅ Resolved | **v1 shot scope**: One-bank shots only (object ball off one rail into pocket). Kick shots deferred to v2. | — |

---

## Version History

| Version | Date | Author | Notes |
|---|---|---|---|
| 0.1 | 2026-07-31 | — | Initial draft |
| 0.2 | 2026-07-31 | — | Added: photo upload input, target ball + pocket output, no number OCR, web app platform (React + FastAPI). Closed Q2, Q4, Q6. |
| 0.3 | 2026-07-31 | — | Narrowed v1 to one-bank shots only. Removed kick shot geometry from v1. Rewrote Feature 2 with ghost-ball contact, object-ball reflection, and pocket intersection. Closed Q7. |
| 0.4 | 2026-07-31 | — | Resolved Q1 (throw in v2), Q3 (no snooker), Q5 (rank shots by ease). Added shot ranking definition to §2.3. All questions resolved. |
| 0.5 | 2026-07-31 | — | Added v2 scope: Feature 3 (one-rail kick shots + diamond system, 2.5-diamond worked example), Feature 4 (cushion throw modelling with empirical correction). TOC updated. |
| 0.6 | 2026-07-31 | — | /grill-me interview: added direct shots to v1 scope, mobile-first result layout, auto table detection confirmed, two-group results panel, DirectShot + BankShot data models, all 6 pockets valid, ghost-ball obstruction confirmed. |

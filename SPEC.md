# Rail-Kick: Pool Table Shot Detection System

## Overview

**Rail-Kick** is a browser-based web application that analyzes a pool/billiards table photograph to:

1. Detect and locate all balls on the table.
2. Identify any **one-rail bank shot** (also called a **one-rail kick shot**) from the cue ball (white ball) to any object ball that can be pocketed.
3. Present the result as an annotated image highlighting the **target ball** and the **target pocket** for each valid shot found.

---

## Table of Contents

- [Glossary](#glossary)
- [System Architecture](#system-architecture)
- [Web Application](#web-application)
- [Feature 1 – Ball Position Analysis](#feature-1--ball-position-analysis)
- [Feature 2 – Rail Bank Shot Detection](#feature-2--rail-bank-shot-detection)
- [Data Models](#data-models)
- [Constraints & Assumptions](#constraints--assumptions)
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
│  • Table boundary detect │             │  • Rail reflection calc  │
│  • Ball detection        │             │  • Obstruction check     │
│  • Ball classification   │             │  • Pocket intersection   │
│    (color/type only,     │             │    test                  │
│     no number OCR)       │             └─────────────────────────┘
│  • Coordinate mapping    │
└──────────────────────────┘
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
2. User uploads a photo of the pool table
   (drag-and-drop area or "Choose File" button).
3. App sends the photo to POST /analyze.
4. Backend processes and returns AnalysisResult JSON.
5. App renders the annotated image with shot overlays.
6. App shows a results panel listing each valid shot:
   "Ball [color/type] → Pocket [id]  via [rail] rail"
7. User can tap/click a shot in the list to highlight
   that specific shot path on the image.
```

### UI Screens

| Screen | Description |
|---|---|
| **Upload Screen** | Full-page drop zone with instructions. Accepts JPG/PNG/WEBP ≤ 20 MB. |
| **Processing Screen** | Spinner / progress indicator while backend analyzes. |
| **Result Screen** | Left: annotated image. Right: scrollable shot list. |
| **Error Screen** | Friendly message if no table is detected or no valid shots exist. |

---

## Feature 1 – Ball Position Analysis

### 1.1 Input

The user uploads a single photo through the web UI.

| Field | Type | Description |
|---|---|---|
| `image` | `multipart/form-data file` | A JPG, PNG, or WEBP photograph of the pool table. Taken from roughly overhead (top-down or slight angle). Maximum file size: 20 MB. |

> **No camera metadata is required.** The perspective correction is computed automatically from the detected table boundary.

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

## Feature 2 – Rail Bank Shot Detection

### 2.1 Concept

For a **one-rail kick shot**, the geometry is modelled using the **mirror / reflection method**:

1. Reflect the object ball's position across the chosen rail.
2. Draw a straight line from the cue ball to the reflected position.
3. The intersection of that line with the rail gives the **contact point**.
4. If the contact point lies within the rail's valid range AND the direct line from contact point to object ball is unobstructed, the shot is valid.

For a **one-rail bank shot**, the same reflection method is applied but:

1. The object ball is the origin of the reflected trajectory.
2. Its path after being struck is reflected across the rail.
3. Check that the reflected path enters a pocket.

### 2.2 Processing Steps

#### Step 1 – Enumerate Rail Candidates

For each rail (`TOP`, `BOTTOM`, `LEFT`, `RIGHT`), generate a candidate shot for each `(cue_ball, object_ball)` pair.

#### Step 2 – Reflection Calculation

Given:
- Cue ball centre: `C = (cx, cy)`
- Object ball centre: `O = (ox, oy)`
- Rail being tested (example: TOP rail at `y = table_height`)

**Mirror the target across the rail:**

```
# Reflect O across the TOP rail (y = table_height)
O_mirror = (ox, 2 * table_height - oy)

# Parameterize the line from C to O_mirror
# Solve for the point P where the line crosses y = table_height
t = (table_height - cy) / (O_mirror.y - cy)
P = (cx + t * (O_mirror.x - cx), table_height)
```

The same logic is applied for all four rails.

#### Step 3 – Contact Point Validity

The contact point `P` must lie **within the cushion's active range** (excluding pocket openings):

| Rail | Valid range |
|---|---|
| TOP / BOTTOM | `x ∈ [pocket_radius, table_width - pocket_radius]`, excluding the side pocket zone at `x ≈ table_width/2` |
| LEFT / RIGHT | `y ∈ [pocket_radius, table_height - pocket_radius]`, excluding the side pocket zone at `y ≈ table_height/2` |

#### Step 4 – Collision / Obstruction Check

Two line segments must be clear of all other balls:

1. **Cue ball → Rail contact point `P`**: No other ball's body intersects this path.
2. **Rail contact point `P` → Object ball**: No other ball's body intersects this path.

**Obstruction test**: For each other ball `B`, compute the perpendicular distance from `B.centre` to the line segment. The path is blocked if that distance is less than `ball_diameter` (57.15 mm).

#### Step 5 – Pocketability Check (One-Rail Bank Only)

After the cue ball strikes the object ball at the computed ghost-ball contact, the object ball travels in a straight line reflected across the rail. Check whether that path intersects any pocket centre within `pocket_radius`. If yes, the shot pockets a ball.

### 2.3 Shot Metadata Output

Each valid shot is emitted as a `Shot` record. The two key user-facing fields are **`object_ball_id`** (which ball to aim for) and **`pocket_id`** (which pocket it enters). Both are always present for a bank shot; for a kick shot `pocket_id` is `null` because the pocket entry depends on a subsequent shot.

**Example – One-Rail Kick Shot:**
```json
{
  "shot_type": "one_rail_kick",
  "cue_ball": { "x": 635, "y": 635 },
  "object_ball_id": "obj1",
  "object_ball_label": "solid-red",
  "rail": "TOP",
  "contact_point": { "x": 980, "y": 1270 },
  "path": [
    { "x": 635,  "y": 635  },
    { "x": 980,  "y": 1270 },
    { "x": 1270, "y": 900  }
  ],
  "estimated_angle_deg": 42.5,
  "obstructed": false,
  "pocket_id": null
}
```

**Example – One-Rail Bank Shot:**
```json
{
  "shot_type": "one_rail_bank",
  "cue_ball": { "x": 635, "y": 635 },
  "object_ball_id": "obj2",
  "object_ball_label": "stripe-blue",
  "rail": "RIGHT",
  "contact_point": { "x": 2540, "y": 480 },
  "path": [
    { "x": 900,  "y": 700  },
    { "x": 2540, "y": 480  },
    { "x": 1270, "y": 0    }
  ],
  "estimated_angle_deg": 35.0,
  "obstructed": false,
  "pocket_id": "BR"
}
```

### 2.4 Annotated Image Output

The backend returns the original photo (perspective-corrected to top-down) with SVG/canvas overlays drawn on it. The frontend renders this image and allows the user to tap a shot in the results panel to toggle highlight of a specific shot path.

| Element | Visual Style | Meaning |
|---|---|---|
| Cue ball | ⚪ White circle outline + label "CUE" | Identified cue ball |
| Object balls | Colour-matched circle outline + `label` text (e.g. "solid-red") | Each object ball — no number shown |
| **Target ball** (selected shot) | 🟡 Bright yellow filled circle | The object ball for the active shot |
| Cue → contact point | 🔵 Blue dashed arrow | First segment of the cue ball's path |
| Contact → object ball (kick) | 🟢 Green dashed arrow | Second segment of kick path |
| Object ball → pocket (bank) | 🟢 Green dashed arrow | Object ball's banking path to pocket |
| Rail contact point | ⬜ White filled dot | Where the ball touches the cushion |
| **Target pocket** (selected shot) | 🟣 Purple pulsing ring | Destination pocket for the active shot |

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

interface Shot {
  shot_type: "one_rail_kick" | "one_rail_bank";
  cue_ball: { x: number; y: number };
  object_ball_id: string;       // key into balls[]
  object_ball_label: string;    // human-readable label, e.g. "solid-red"
  rail: RailId;
  contact_point: { x: number; y: number };
  path: Array<{ x: number; y: number }>;  // ordered waypoints
  estimated_angle_deg: number;            // angle at rail contact
  obstructed: boolean;
  pocket_id: string | null;     // pocket id for bank shots; null for kick shots
}

interface AnalysisResult {
  table_dims_mm: { width: number; height: number };
  pockets: Pocket[];
  balls: Ball[];
  shots: Shot[];                // all valid one-rail shots found
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
| C5 | Only **one-rail** kick and bank shots are detected. Multi-rail combinations are out of scope for v1. |
| C6 | Ball cushion compression, throw, squirt, and spin are **not modelled** in v1; pure geometric reflection is used. |
| C7 | **Ball number recognition is out of scope.** Balls are identified by color/type only. |
| C8 | The cue ball must be visually distinguishable (white or near-white). |
| C9 | A minimum of 2 balls (cue ball + at least one object ball) must be present on the table. |
| C10 | Balls must not be completely overlapping in the image. |
| C11 | The application runs in a **web browser**. No native app installation is required. |

---

## Open Questions

| # | Status | Question | Impact |
|---|---|---|---|
| Q1 | 🔴 Open | Should **cushion throw** or ghost-ball aiming offset be modelled in v2? | Affects geometric accuracy for cut shots off the rail. |
| Q2 | ✅ Resolved | Input is a **photo uploaded via the web UI** (drag-and-drop or file picker). Live video is out of scope for v1. | — |
| Q3 | 🔴 Open | Should the system support **snooker** tables in a future version? | Requires a separate ball classifier and pocket geometry. |
| Q4 | ✅ Resolved | **Number recognition is not required.** Color/type label is sufficient. | — |
| Q5 | 🔴 Open | Should the output **rank shots** (easiest angle first) or list all equally? | Affects UX design of the results panel. |
| Q6 | ✅ Resolved | Target platform is **web browser** (React SPA + Python backend). | Stack selected: Vite + FastAPI + OpenCV. |

---

## Version History

| Version | Date | Author | Notes |
|---|---|---|---|
| 0.1 | 2026-07-31 | — | Initial draft |
| 0.2 | 2026-07-31 | — | Added: photo upload input, target ball + pocket output, no number OCR, web app platform (React + FastAPI). Closed Q2, Q4, Q6. |

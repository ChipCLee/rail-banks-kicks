# Rail-Kick: Pool Table Shot Detection System

**Rail-Kick** is a browser-based web application that analyzes pool/billiards table photographs to detect and calculate **direct shots**, **one-bank shots**, and **one-rail kick shots** with diamond system reference points and cushion throw corrections.

---

## Features

- **Direct Shot Detection (v1)**: Identifies straight-line pocketable shots from cue ball to object ball into any of the 6 pockets.
- **One-Bank Shot Detection (v1)**: Calculates object ball reflection trajectories off all 4 rails into target pockets, sorted by ease score ($|\text{angle} - 90^\circ|$).
- **One-Rail Kick Shot Detection (v2)**: Calculates cue ball kick trajectories off rail cushions to strike object balls, formatted with **diamond marker labels** (e.g. `2.5 diamonds from TL on TOP rail`).
- **Cushion Throw Correction (v2)**: Applies empirical friction correction to rebound angles at shallow impact angles.
- **YOLOv8-small Vision Pipeline**: Custom YOLOv8s segmentation detects the table and balls, OpenCV performs homography/metric mapping, and automatic device selection uses NVIDIA CUDA, Apple MPS, or CPU.
- **iPhone Camera Input**: Applies EXIF orientation, accepts HEIC/HEIF as well as JPEG/PNG/WEBP, and bounds high-resolution camera images before inference.
- **Mobile-First Responsive UI**: Interactive top-down annotated table view with tap-to-highlight shot visualization.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 React SPA Frontend (Port 3000)               │
│  • Vite + React + Vanilla CSS (Dark Theme)                  │
│  • Mobile-first stacked layout                              │
│  • Photo upload drag & drop + camera picker                 │
└──────────────────────────────┬──────────────────────────────┘
                               │  multipart/form-data (image)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                Python FastAPI Backend (Port 8000)           │
│  • Custom YOLOv8s segmentation on CUDA / MPS / CPU          │
│  • OpenCV perspective correction and metric mapping         │
│  • Vector geometry & reflection physics engine              │
│  • Top-down annotated image renderer (Base64 JPEG)          │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start (Docker Compose)

The default Compose stack runs the portable CPU backend. Copy the validated custom
checkpoint to `backend/weights/rail_kick_yolov8s_seg.pt` before starting it:

```bash
# Clone the repository
git clone https://github.com/ChipCLee/rail-banks-kicks.git
cd rail-kick

# Build and start services
docker-compose up --build
```

- **Web Frontend**: [http://localhost:3000](http://localhost:3000)
- **Backend API**: [http://localhost:8000](http://localhost:8000)
- **API Docs (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Local Development Setup

### 1. Backend Setup (FastAPI + YOLOv8s)

**Prerequisite**: `uv` installed. The backend pins Python 3.11 in
`backend/.python-version`; `uv` creates and manages the project environment.

```bash
cd backend

# Create/update .venv from the committed lockfile
uv sync --frozen

# Required custom checkpoint (stock COCO weights are not compatible)
export YOLO_MODEL_PATH="$PWD/weights/rail_kick_yolov8s_seg.pt"

# Run backend development server
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Device selection defaults to NVIDIA CUDA, then Apple MPS, then CPU. Override it
with `CV_DEVICE=auto|cuda|mps|cpu`. On Apple Silicon, run the backend natively;
Linux containers under Docker Desktop do not expose Metal/MPS.

For an NVIDIA host with the NVIDIA Container Toolkit:

```bash
docker compose -f docker-compose.yml -f docker-compose.nvidia.yml up --build
```

The custom checkpoint is trained from `yolov8s-seg.pt` using the mutually
exclusive classes `table`, `cue_ball`, `eight_ball`, and `object_ball`:

```bash
cd backend
uv run python train_yolo.py --data training/rail_kick.yaml --device mps  # Apple Silicon
uv run python train_yolo.py --data training/rail_kick.yaml --device 0    # NVIDIA
```

See [TRAIN.md](TRAIN.md) for dataset layout, segmentation rules, validation
gates, checkpoint promotion, and troubleshooting.

#### Running Backend Unit Tests

```bash
cd backend
uv run python -m unittest discover -p "test_*.py"
```

---

### 2. Frontend Setup (React + Vite)

**Prerequisites**: Node.js v18+ & npm installed.

```bash
cd frontend

# Install dependencies
npm install

# Start Vite dev server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## API Specification

### `POST /analyze`

Upload a pool table image for analysis.

- **Content-Type**: `multipart/form-data`
- **Body**: `image` (JPG, PNG, WEBP, HEIC, or HEIF file, $\le 20\text{ MB}$)

#### Response (`200 OK`)

```json
{
  "table_dims_mm": { "width": 2540.0, "height": 1270.0 },
  "pockets": [
    { "id": "TL", "x": 0.0, "y": 1270.0, "radius_mm": 57.0 },
    { "id": "ML", "x": 0.0, "y": 635.0, "radius_mm": 63.0 }
  ],
  "balls": [
    { "id": "cue", "label": "cue", "x": 400.0, "y": 900.0, "radius_mm": 28.575 },
    { "id": "obj1", "label": "eight", "x": 1800.0, "y": 635.0, "radius_mm": 28.575 }
  ],
  "direct_shots": [ ... ],
  "bank_shots": [
    {
      "shot_type": "one_bank",
      "cue_ball": { "x": 800.0, "y": 980.8 },
      "object_ball_id": "obj1",
      "object_ball_label": "eight",
      "rail": "RIGHT",
      "contact_point": { "x": 2540.0, "y": 840.2 },
      "path": [
        { "x": 800.0, "y": 980.8 },
        { "x": 1800.0, "y": 900.0 },
        { "x": 2540.0, "y": 840.2 },
        { "x": 0.0, "y": 635.0 }
      ],
      "bank_angle_deg": 38.0,
      "throw_correction_deg": 3.94,
      "adjusted_rebound_angle_deg": 34.06,
      "ease_score": 52.0,
      "pocket_id": "ML"
    }
  ],
  "kick_shots": [ ... ],
  "annotated_image_b64": "<base64_encoded_jpeg>"
}
```

---

## Project Structure

```
rail-kick/
├── SPEC.md                      # Technical specification (v0.7)
├── TRAIN.md                     # Custom YOLOv8s dataset and training guide
├── E2E_TEST_CASES.md            # Comprehensive E2E test suite (37 test cases)
├── docker-compose.yml           # Docker orchestration file
├── images/                      # Source camera photographs
│   ├── IMG_5305.HEIC            # iPhone HEIC training source
│   ├── IMG_5306.HEIC            # iPhone HEIC training source
│   ├── IMG_5307.HEIC            # iPhone HEIC training source
│   ├── example.jpg              # iPhone landscape table example
│   └── example_1.jpg            # iPhone portrait table example
├── backend/
│   ├── main.py                  # FastAPI REST endpoints
│   ├── models.py                # Pydantic data schemas
│   ├── geometry.py              # Direct & bank shot vector geometry
│   ├── cv_module.py             # YOLOv8s inference and OpenCV metric mapping
│   ├── image_input.py           # EXIF/HEIC camera image normalization
│   ├── train_yolo.py            # Custom segmentation training entry point
│   ├── edit_yolo_pairs.py       # Interactive image/segmentation-label editor
│   ├── training/                # Dataset schema
│   ├── weights/                 # Runtime custom checkpoint mount point
│   ├── v2_kick_shots.py         # Kick shot detection & diamond calculation
│   ├── cushion_throw.py         # Cushion throw empirical model
│   ├── annotate.py              # Top-down visual overlay renderer
│   ├── pyproject.toml           # Python project and dependency declarations
│   ├── uv.lock                  # Cross-platform locked Python dependencies
│   ├── .python-version          # uv-managed Python version
│   ├── Dockerfile               # Backend Docker container
│   ├── Dockerfile.nvidia        # NVIDIA CUDA backend container
│   ├── test_geometry.py         # Geometry unit tests
│   ├── test_v2_kick.py          # Kick shot unit tests
│   └── test_v2_throw.py         # Cushion throw unit tests
└── frontend/
    ├── src/
    │   ├── App.jsx              # Application state manager
    │   ├── api.js               # REST API client
    │   ├── index.css            # Custom CSS design system
    │   └── components/
    │       ├── UploadScreen.jsx     # Drag & drop photo upload
    │       ├── ProcessingScreen.jsx # Spinner screen
    │       ├── ResultScreen.jsx     # Mobile-first result viewer
    │       ├── ShotList.jsx         # Grouped & ranked shot list
    │       └── ErrorScreen.jsx      # Friendly error display
    ├── package.json
    ├── vite.config.js
    └── Dockerfile
```

---

## License

Rail-Kick source code is MIT licensed. Ultralytics and model artifacts have their
own license terms; confirm that the Ultralytics license selected for training and
deployment is compatible with the intended distribution.

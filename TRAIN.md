# Training `rail_kick_yolov8s_seg.pt`

This guide produces the custom YOLOv8-small segmentation checkpoint required by
Rail-Kick. The finished file must be installed as:

```text
backend/weights/rail_kick_yolov8s_seg.pt
```

Stock `yolov8s-seg.pt` is only the training starting point. It cannot be used as
the application checkpoint because its COCO classes do not include the
Rail-Kick table and ball classes.

## 1. Hardware and software

Supported training devices:

- Apple Silicon Mac: PyTorch MPS, selected with `--device mps`.
- NVIDIA GPU: CUDA, selected with `--device 0` for the first GPU.
- CPU: supported with `--device cpu`, but full training will be very slow.

Install `uv`, then create the locked backend environment. Do not create or
activate a virtual environment manually; `uv` manages `backend/.venv` and the
Python 3.11 interpreter declared in `.python-version`.

```bash
cd backend
uv sync --frozen
```

Dependencies are declared in `pyproject.toml` and resolved in `uv.lock`. Use
`uv add <package>` or `uv add --dev <package>` when changing dependencies, then
commit both files. Do not manage this project with `pip install` or a handwritten
requirements file.

Confirm the accelerator before preparing a long run.

Apple Silicon:

```bash
uv run python - <<'PY'
import torch
print("MPS available:", torch.backends.mps.is_available())
PY
```

NVIDIA:

```bash
nvidia-smi
uv run python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())
PY
```

If the requested accelerator prints `False`, fix the PyTorch installation before
starting training. Do not silently train a large run on CPU.

## 2. Collect the image dataset

Use real pool-table photographs, with an emphasis on iPhone 16 camera images.
Include the conditions the production application will receive:

- Portrait and landscape orientation.
- HEIC and JPEG originals.
- 1× and wide-angle captures.
- Blue, green, and red felt.
- 7-foot, 8-foot, and 9-foot tables.
- Bright, dim, uneven, and reflective lighting.
- Light and dark rails, varied room backgrounds, and people near the table.
- Balls near cushions and pockets, touching balls, clusters, and sparse layouts.
- Empty tables and difficult but valid full-table views.
- Negative images with no pool table for false-positive evaluation.

The full playable surface and all six pockets should normally be visible. Keep
the original resolution and EXIF metadata in the source archive. Annotation
exports may be converted to JPEG or PNG.

Avoid collecting many nearly identical frames from one burst. Near-duplicate
images make validation scores look better without improving real-world quality.

## 3. Annotate segmentation masks

For a complete one-image example derived from
[`images/example.jpg`](images/example.jpg), see
[`backend/datasets/rail_kick_example`](backend/datasets/rail_kick_example). The
educational dataset contains its own copy at `images/train/example.jpg`, paired
with a five-instance polygon label, plus the YAML at
`backend/training/rail_kick_example.yaml`.

Use a tool capable of exporting Ultralytics YOLO segmentation labels. Create
these mutually exclusive classes with the exact spelling and order:

| ID | Class | Annotation rule |
|---:|---|---|
| 0 | `table` | Polygon around the playable felt boundary. Do not include outer wooden rails or room background. |
| 1 | `cue_ball` | Visible outline of the white cue ball, including marked/measles cue balls. |
| 2 | `eight_ball` | Visible outline of the black eight ball. |
| 3 | `object_ball` | Visible outline of every other pool ball, whether solid or striped. |

Labeling rules:

1. Draw a separate polygon for every visible ball.
2. Trace only visible pixels when a ball is partially occluded.
3. Keep pocket openings out of the `table` mask.
4. Do not label reflections, printed balls, lamps, or circular background objects.
5. Do not label snooker balls or carom tables unless support is intentionally added.
6. Check class IDs after every export; changing class order invalidates the checkpoint contract.
7. Include an empty `.txt` label file for a negative image with no target objects.

Each YOLO segmentation label line has this form:

```text
class_id x1 y1 x2 y2 x3 y3 ...
```

Coordinates are normalized to `0.0–1.0`. A polygon needs at least three points.
The annotation tool should generate these coordinates; manual label-file editing
is discouraged.

## 4. Split by table and photo session

Use approximately:

- 70% training
- 20% validation
- 10% final test

Split by physical table and photo session, not by randomly distributing adjacent
photos. Images of the same table captured in the same lighting session must stay
in one split. Otherwise the validation metrics will be misleading.

Do not tune thresholds against the final test split. Use it only for checkpoint
promotion.

## 5. Create the dataset structure

From the repository root, arrange the export as:

```text
backend/
├── datasets/
│   └── rail_kick/
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── labels/
│           ├── train/
│           ├── val/
│           └── test/
├── training/
│   └── rail_kick.yaml
└── train_yolo.py
```

Every image must have a same-stem label file:

```text
images/train/table_session_001.jpg
labels/train/table_session_001.txt
```

The checked-in dataset definition is
[`backend/training/rail_kick.yaml`](backend/training/rail_kick.yaml):

```yaml
path: datasets/rail_kick
train: images/train
val: images/val
test: images/test

names:
  0: table
  1: cue_ball
  2: eight_ball
  3: object_ball
```

Run training from `backend/` so the documented dataset and output paths stay
consistent. If the local Ultralytics configuration resolves datasets elsewhere,
replace `path` with the absolute path to `backend/datasets/rail_kick`.

Before training, verify:

- No missing image/label pairs.
- Every class ID is between 0 and 3.
- Every polygon coordinate is between 0 and 1.
- No polygon has fewer than three points.
- Validation and test sessions do not appear in training.
- A sample of masks visually follows the felt and ball boundaries.

### View and edit image/label pairs

Start the local dataset editor:

```bash
cd backend
uv run python edit_yolo_pairs.py \
  --dataset datasets/rail_kick \
  --split train \
  --data training/rail_kick.yaml \
  --open
```

The editor pairs files by filename stem and reports structural label problems.
Select a polygon to drag its vertices, or choose **Add polygon**, select its
class, click at least three boundary points, and finish it. **Save label** writes
the normalized YOLO polygon data back to the paired `.txt` file.

You can change the dataset root, split, and class YAML at the top of the page.
To start another dataset, expand **Create a new dataset from selected images**,
choose the source photos, enter a new dataset path, and create it. The tool adds
the standard `images/{train,val,test}` and `labels/{train,val,test}` directories,
an empty label for each selected image, and a starter `dataset.yaml`.

To open the checked-in educational example:

```bash
uv run python edit_yolo_pairs.py \
  --data training/rail_kick_example.yaml \
  --open
```

For a read-only file that can be shared or archived, add
`--export datasets/rail_kick/viewer-train.html --strict`. The generated HTML
references images by relative path, so keep it inside or near the dataset.
Browser support for HEIC varies; use JPEG or PNG annotation exports when a
browser cannot display an HEIC source.

## 6. Run a short pipeline check

Start with a small run to catch path, label, and memory problems:

Apple Silicon:

```bash
cd backend
uv run python train_yolo.py \
  --data training/rail_kick.yaml \
  --device mps \
  --epochs 3 \
  --imgsz 640 \
  --project runs/rail_kick_smoke
```

NVIDIA:

```bash
cd backend
uv run python train_yolo.py \
  --data training/rail_kick.yaml \
  --device 0 \
  --epochs 3 \
  --imgsz 640 \
  --project runs/rail_kick_smoke
```

Inspect the generated label previews and training plots. Do not start the full
run if masks are shifted, class colors are wrong, or validation images are not
being found.

## 7. Train the full model

The repository defaults are 150 epochs and 1280-pixel inference/training size.
The training script starts from `yolov8s-seg.pt` and applies rotation,
perspective, and flip augmentation suitable for overhead table photographs.

Apple Silicon:

```bash
cd backend
uv run python train_yolo.py \
  --data training/rail_kick.yaml \
  --device mps \
  --epochs 150 \
  --imgsz 1280 \
  --project runs/rail_kick
```

NVIDIA GPU 0:

```bash
cd backend
uv run python train_yolo.py \
  --data training/rail_kick.yaml \
  --device 0 \
  --epochs 150 \
  --imgsz 1280 \
  --project runs/rail_kick
```

Expected run directory:

```text
backend/runs/rail_kick/yolov8s_seg/
```

Important artifacts include:

```text
weights/best.pt       validation-selected checkpoint
weights/last.pt       final-epoch checkpoint
results.csv           epoch metrics
results.png           training curves
confusion_matrix.png  class confusion summary
val_batch*_pred.jpg   visual validation predictions
```

Use `best.pt`, not `last.pt`, for evaluation and deployment.

If the GPU runs out of memory, reduce `--imgsz` to 1024 or 960. Small billiard
balls benefit from resolution, so reduce image size only as far as necessary.
The checked-in script lets Ultralytics choose its default batch size; if explicit
batch control becomes necessary, add a `--batch` option to the script and record
the value with the experiment.

## 8. Validate the candidate checkpoint

Training loss alone is not an acceptance criterion. Inspect both segmentation
metrics and application-level millimetre accuracy on held-out iPhone images.

The release gates from `SPEC.md` are:

- Table localization success at least 98%.
- Ball recall at least 95%.
- Ball precision at least 98%.
- Median ball-centre error no greater than 15 mm after homography.
- No emitted ball centre outside the one-ball-radius playfield inset.
- Smoke tests pass on both Apple MPS and NVIDIA CUDA.

Also review failures by capture condition: orientation, lens choice, felt color,
lighting, table size, ball-to-rail distance, and ball clustering. A single overall
score can hide a weak subgroup.

To run an additional Ultralytics validation pass from `backend/`:

```bash
uv run python - <<'PY'
from ultralytics import YOLO

model = YOLO("runs/rail_kick/yolov8s_seg/weights/best.pt", task="segment")
metrics = model.val(data="training/rail_kick.yaml", imgsz=1280, device="mps")
print(metrics)
PY
```

Change `device="mps"` to `device=0` on NVIDIA or `device="cpu"` for a CPU check.

## 9. Install the production checkpoint

After the candidate passes all validation gates:

```bash
cd backend
cp runs/rail_kick/yolov8s_seg/weights/best.pt \
   weights/rail_kick_yolov8s_seg.pt
```

The `.pt` file is intentionally ignored by Git. Store the validated checkpoint in
the project's approved artifact storage and copy or mount it during deployment.
Record at least:

- Model checksum.
- Training commit.
- Dataset version and split manifest.
- Training command and device.
- Ultralytics and PyTorch versions.
- Validation metrics and failed-image review.

Generate a checksum:

```bash
shasum -a 256 weights/rail_kick_yolov8s_seg.pt
```

## 10. Run Rail-Kick with the checkpoint

Native Apple Silicon:

```bash
cd backend
export CV_DEVICE=mps
export YOLO_MODEL_PATH="$PWD/weights/rail_kick_yolov8s_seg.pt"
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

Native NVIDIA:

```bash
cd backend
export CV_DEVICE=cuda
export YOLO_MODEL_PATH="$PWD/weights/rail_kick_yolov8s_seg.pt"
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

NVIDIA Docker:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.nvidia.yml \
  up --build
```

Apple MPS is not exposed to the Linux backend container. Run the backend natively
on macOS when Apple GPU acceleration is required.

Check model readiness:

```bash
curl http://localhost:8000/health
```

A ready response reports `weights_available: true`, `loaded: true`, and the
selected device. Then submit several held-out photographs through the web UI and
inspect ball positions and rendered paths.

## 11. Run regression tests

The regular test suite injects deterministic predictions and therefore does not
measure checkpoint accuracy. It verifies integration, coordinate conversion,
device selection, API behavior, and rendering:

```bash
cd backend
uv run python -m unittest discover -p "test_*.py" -v
```

Run the real checkpoint validation separately before every promotion. Never treat
the deterministic E2E test as evidence that a newly trained model is accurate.

## 12. Troubleshooting

### `Custom YOLOv8s weights were not found`

Confirm the file exists and use an absolute runtime path:

```bash
ls -lh backend/weights/rail_kick_yolov8s_seg.pt
export YOLO_MODEL_PATH="$(pwd)/backend/weights/rail_kick_yolov8s_seg.pt"
```

### `checkpoint is not a Rail-Kick model`

The checkpoint does not contain both the `table` class and at least one supported
ball class. Verify the dataset YAML class names and make sure `best.pt` came from
this training configuration rather than an unmodified COCO model.

### Dataset images are not found

Run from `backend/`, verify the directory tree in section 5, or change the YAML
`path` to an absolute path.

### MPS is unavailable

Confirm the machine is Apple Silicon, macOS and PyTorch support MPS, and the
virtual environment is using the intended Python interpreter. Use
`CV_DEVICE=cpu` only for diagnosis.

### CUDA is unavailable

Check `nvidia-smi`, the NVIDIA driver, the CUDA-enabled PyTorch build, and—in a
container—the NVIDIA Container Toolkit. Confirm the container was started with
GPU access.

### Out-of-memory during training

Reduce image size gradually from 1280 to 1024 or 960, close other GPU workloads,
and retry the smoke run before restarting a full experiment.

### Tables work but balls are missed

Increase close-to-rail, clustered, distant-camera, and difficult-lighting ball
examples. Check annotation polygons at full resolution. Do not compensate only
by lowering inference confidence, because that can introduce pocket and
reflection false positives.

### Good model metrics but poor millimetre coordinates

Inspect the `table` mask corners. Ball metrics can be good while a loose or
rounded table mask distorts the homography. Add difficult rail/pocket boundaries
and evaluate table-corner error separately.

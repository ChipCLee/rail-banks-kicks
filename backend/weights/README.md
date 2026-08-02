# Model weights

Place the validated custom YOLOv8-small segmentation checkpoint here as:

`rail_kick_yolov8s_seg.pt`

The checkpoint is intentionally not downloaded automatically. Stock COCO weights
do not define the Rail-Kick `table`, `cue_ball`, `eight_ball`, and `object_ball`
classes. Override the location with `YOLO_MODEL_PATH`.

Train a checkpoint with:

```bash
uv run python train_yolo.py --data training/rail_kick.yaml --device mps
# or use --device 0 on NVIDIA CUDA
```

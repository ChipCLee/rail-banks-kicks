"""Fine-tune the custom Rail-Kick YOLOv8-small segmentation model."""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="training/rail_kick.yaml")
    parser.add_argument("--base", default="yolov8s-seg.pt")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="")
    parser.add_argument("--project", default="runs/rail_kick")
    args = parser.parse_args()

    model = YOLO(args.base, task="segment")
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device or None,
        project=args.project,
        name="yolov8s_seg",
        degrees=180,
        perspective=0.001,
        fliplr=0.5,
        flipud=0.5,
    )

    best = Path(args.project) / "yolov8s_seg" / "weights" / "best.pt"
    print(f"Copy the validated weights from {best} to weights/rail_kick_yolov8s_seg.pt")


if __name__ == "__main__":
    main()

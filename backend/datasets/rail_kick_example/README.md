# One-image annotation example

This directory demonstrates the file pairing and polygon format expected by
Ultralytics YOLO instance segmentation. It is educational data, not a dataset
large enough to train a useful model.

Its source photograph is the repository fixture
[`images/example.jpg`](../../../images/example.jpg). This directory keeps a
separate copy because YOLO datasets require images and labels in matching split
directories.

```text
images/train/example.jpg
labels/train/example.txt
```

The five rows in `example.txt` represent:

1. Class `0`: one simplified quadrilateral around the playable table surface.
2. Class `1`: the white cue ball.
3. Class `3`: the blue-striped object ball.
4. Class `3`: the central red-striped object ball.
5. Class `3`: the red object ball near the side pocket.

There is no class `2` row because the photograph has no eight ball. The ball
outlines use 16-point circular polygons. The table polygon is deliberately
simple so its normalized coordinates are easy to inspect; production labels
should trace the visible cushion boundary and pocket jaws more carefully.

All coordinates are normalized:

```text
normalized_x = pixel_x / image_width
normalized_y = pixel_y / image_height
```

For this 2643 × 2641 image, the cue-ball center is approximately `(391, 943)`,
which becomes `(0.148, 0.357)` after normalization.

Open the image and label together in an annotation tool to visualize and refine
the polygons. Do not add this example to production training until its masks
have been manually reviewed.

Open `annotation_preview.svg` for an immediate overlay: green is class `0`
(`table`), yellow is class `1` (`cue_ball`), and red is class `3`
(`object_ball`). The preview uses SVG circles for readability; `example.txt`
stores those ball outlines as 16-point segmentation polygons.

Open the dataset editor from `backend/` with:

```bash
uv run python edit_yolo_pairs.py --data training/rail_kick_example.yaml --open
```

This opens the local editor with image/label navigation, polygon creation,
vertex editing, saving, and structural validation results.

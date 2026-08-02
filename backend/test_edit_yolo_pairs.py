import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from edit_yolo_pairs import DEFAULT_NAMES, create_editor_app, parse_names_yaml, render_viewer


class TestYoloPairViewer(unittest.TestCase):
    def test_generates_overlay_and_pair_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            labels = root / "labels"
            images.mkdir()
            labels.mkdir()
            Image.new("RGB", (100, 50), "blue").save(images / "sample.jpg")
            (labels / "sample.txt").write_text(
                "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n"
                "1 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6\n",
                encoding="utf-8",
            )
            output = root / "viewer.html"

            report = render_viewer(images, labels, output)
            document = output.read_text(encoding="utf-8")

            self.assertEqual(report.pairs, 1)
            self.assertEqual(report.instances, 2)
            self.assertFalse(report.errors)
            self.assertIn("sample.jpg", document)
            self.assertIn("0 table", document)
            self.assertIn("1 cue_ball", document)
            self.assertIn("<polygon", document)

    def test_reports_detection_box_as_invalid_segmentation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            labels = root / "labels"
            images.mkdir()
            labels.mkdir()
            Image.new("RGB", (20, 20), "black").save(images / "bad.png")
            (labels / "bad.txt").write_text("3 0.5 0.5 0.2 0.2\n", encoding="utf-8")

            report = render_viewer(images, labels, root / "viewer.html")
            self.assertEqual(len(report.errors), 1)
            self.assertIn("at least 3 x/y polygon pairs", report.errors[0])

    def test_reads_numeric_names_from_dataset_yaml(self):
        with tempfile.TemporaryDirectory() as temporary:
            yaml_path = Path(temporary) / "dataset.yaml"
            yaml_path.write_text(
                "path: dataset\nnames:\n  0: table\n  1: cue_ball\n",
                encoding="utf-8",
            )
            self.assertEqual(parse_names_yaml(yaml_path), {0: "table", 1: "cue_ball"})
            self.assertEqual(DEFAULT_NAMES[3], "object_ball")


class TestYoloPairEditor(unittest.TestCase):
    def _dataset(self, root: Path) -> Path:
        dataset = root / "dataset"
        images = dataset / "images" / "train"
        labels = dataset / "labels" / "train"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        Image.new("RGB", (100, 50), "green").save(images / "sample.jpg")
        (labels / "sample.txt").write_text(
            "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n",
            encoding="utf-8",
        )
        (dataset / "dataset.yaml").write_text(
            "path: .\ntrain: images/train\nval: images/val\n\nnames:\n  0: table\n  1: cue_ball\n",
            encoding="utf-8",
        )
        return dataset

    def test_loads_dataset_and_editor_page(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = self._dataset(Path(temporary))
            client = TestClient(create_editor_app(dataset))

            page = client.get("/")
            response = client.get("/api/dataset", params={"path": str(dataset), "split": "train"})

            self.assertEqual(page.status_code, 200)
            self.assertIn("Create a new dataset", page.text)
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["pairs"][0]["image_name"], "sample.jpg")
            self.assertEqual(payload["pairs"][0]["instances"][0]["class_id"], 0)

    def test_saves_edited_polygon(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = self._dataset(Path(temporary))
            label = dataset / "labels" / "train" / "sample.txt"
            client = TestClient(create_editor_app(dataset))

            response = client.post(
                "/api/labels/save",
                json={
                    "label_path": str(label),
                    "instances": [{"class_id": 1, "points": [[0.2, 0.2], [0.8, 0.2], [0.5, 0.8]]}],
                },
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(label.read_text(encoding="utf-8"), "1 0.200000 0.200000 0.800000 0.200000 0.500000 0.800000\n")

    def test_creates_dataset_from_uploaded_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jpg"
            Image.new("RGB", (24, 16), "orange").save(source)
            dataset = root / "new_dataset"
            client = TestClient(create_editor_app(root / "unused"))

            with source.open("rb") as image_file:
                response = client.post(
                    "/api/datasets/create",
                    data={"dataset_path": str(dataset), "split": "train"},
                    files={"files": ("example.jpg", image_file, "image/jpeg")},
                )

            self.assertEqual(response.status_code, 200)
            self.assertTrue((dataset / "images" / "train" / "example.jpg").is_file())
            self.assertEqual((dataset / "labels" / "train" / "example.txt").read_text(), "")
            self.assertTrue((dataset / "images" / "val").is_dir())
            self.assertIn('path: "', (dataset / "dataset.yaml").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

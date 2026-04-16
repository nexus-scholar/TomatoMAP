import json
import sys
import tempfile
import unittest
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from src.experiments.split_validation import validate_manifest_against_files


class TestSplitValidation(unittest.TestCase):
    def test_manifest_validation_passes_on_consistent_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            coco_dir = base / "coco"
            images_dir = base / "images"
            labels_dir = base / "labels"
            coco_dir.mkdir(parents=True)
            images_dir.mkdir(parents=True)
            labels_dir.mkdir(parents=True)

            split_to_files = {
                "train": ["a.JPG", "b.JPG"],
                "val": ["c.JPG"],
                "test": ["d.JPG"],
            }

            for split, files in split_to_files.items():
                payload = {
                    "images": [{"file_name": name, "id": i + 1} for i, name in enumerate(files)],
                    "annotations": [],
                    "categories": [{"id": 1, "name": "tomato", "supercategory": "none"}],
                }
                with open(coco_dir / f"{split}.json", "w", encoding="utf-8") as f:
                    json.dump(payload, f)

            for files in split_to_files.values():
                for name in files:
                    (images_dir / name).write_bytes(b"img")
                    with open(labels_dir / f"{Path(name).stem}.json", "w", encoding="utf-8") as f:
                        json.dump({"info": {"name": name}, "objects": []}, f)

            manifest = {
                "frozen": True,
                "splits": split_to_files,
            }

            summary = validate_manifest_against_files(manifest, coco_dir, images_dir, labels_dir)
            self.assertEqual(summary["counts"]["train"], 2)
            self.assertEqual(summary["counts"]["val"], 1)
            self.assertEqual(summary["counts"]["test"], 1)


if __name__ == "__main__":
    unittest.main()


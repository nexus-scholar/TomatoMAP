import json
import sys
import tempfile
import unittest
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from src.data.isat_to_coco import convert_isat_folder_to_coco


class TestIsatToCoco(unittest.TestCase):
    def test_deterministic_split_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            task_dir = base / "images"
            lbl_dir = base / "labels"
            out_a = base / "out_a"
            out_b = base / "out_b"
            categories = base / "isat.yaml"

            task_dir.mkdir(parents=True)
            lbl_dir.mkdir(parents=True)

            categories.write_text("label:\n  - name: __background__\n  - name: tomato\n", encoding="utf-8")

            for i in range(6):
                stem = f"img_{i}"
                (task_dir / f"{stem}.JPG").write_bytes(b"img")
                payload = {
                    "info": {"name": f"{stem}.JPG", "width": 100, "height": 100},
                    "objects": [
                        {
                            "category": "tomato",
                            "segmentation": [[1, 1], [10, 1], [10, 10]],
                            "bbox": [1, 1, 9, 9],
                            "area": 81,
                        }
                    ],
                }
                with open(lbl_dir / f"{stem}.json", "w", encoding="utf-8") as f:
                    json.dump(payload, f)

            result_a = convert_isat_folder_to_coco(
                task_dir=str(task_dir),
                label_dir=str(lbl_dir),
                categories_path=str(categories),
                output_dir=str(out_a),
                train_ratio=0.5,
                val_ratio=0.25,
                seed=42,
            )
            result_b = convert_isat_folder_to_coco(
                task_dir=str(task_dir),
                label_dir=str(lbl_dir),
                categories_path=str(categories),
                output_dir=str(out_b),
                train_ratio=0.5,
                val_ratio=0.25,
                seed=42,
            )

            self.assertEqual(result_a["split_counts"], result_b["split_counts"])

            total_annotations = 0
            tomato_category_ids: set[int] = set()

            for split in ("train", "val", "test"):
                with open(out_a / f"{split}.json", "r", encoding="utf-8") as fa:
                    a = json.load(fa)
                with open(out_b / f"{split}.json", "r", encoding="utf-8") as fb:
                    b = json.load(fb)

                self.assertEqual([x["file_name"] for x in a["images"]], [x["file_name"] for x in b["images"]])

                categories_by_name = {c["name"]: c["id"] for c in a.get("categories", [])}
                self.assertIn("tomato", categories_by_name)
                tomato_id = categories_by_name["tomato"]
                tomato_category_ids.add(tomato_id)

                annotations = a.get("annotations", [])
                total_annotations += len(annotations)
                for ann in annotations:
                    self.assertEqual(ann["category_id"], tomato_id)

            self.assertGreater(total_annotations, 0)
            self.assertEqual(len(tomato_category_ids), 1)


if __name__ == "__main__":
    unittest.main()

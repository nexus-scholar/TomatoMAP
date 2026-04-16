import json
import sys
import tempfile
import unittest
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from src.data.seg_audit import audit_segmentation


class TestSegAudit(unittest.TestCase):
    def test_segmentation_image_label_matching(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            img_dir = base / "images"
            lbl_dir = base / "labels"
            img_dir.mkdir(parents=True)
            lbl_dir.mkdir(parents=True)

            (img_dir / "a.JPG").write_bytes(b"x")
            (img_dir / "b.JPG").write_bytes(b"x")

            with open(lbl_dir / "a.json", "w", encoding="utf-8") as f:
                json.dump({"info": {"name": "a.JPG"}, "objects": []}, f)

            result = audit_segmentation(str(img_dir), str(lbl_dir))

            self.assertEqual(result["total_images"], 2)
            self.assertEqual(result["total_labels"], 1)
            self.assertEqual(result["matched_images"], 1)
            self.assertEqual(result["missing_count"], 1)
            self.assertEqual(result["missing_labels"], ["b"])


if __name__ == "__main__":
    unittest.main()

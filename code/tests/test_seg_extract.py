import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from src.data.seg_extract import extract_segmentation_labels


class TestSegExtract(unittest.TestCase):
    def test_extract_json_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            zip_path = base / "labels.zip"
            out_dir = base / "out"

            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("nested/a.json", json.dumps({"k": 1}))
                zf.writestr("nested/b.JSON", json.dumps({"k": 2}))
                zf.writestr("nested/c.txt", "ignore")

            result = extract_segmentation_labels(str(zip_path), str(out_dir), dry_run=False)

            self.assertEqual(result["json_in_zip"], 2)
            self.assertEqual(result["extracted"], 2)
            self.assertTrue((out_dir / "a.json").exists())
            self.assertTrue((out_dir / "b.JSON").exists())


if __name__ == "__main__":
    unittest.main()

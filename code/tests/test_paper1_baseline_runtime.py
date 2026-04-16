import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from src.experiments.paper1_baseline import _prepare_dataset_view


class TestPaper1BaselineRuntime(unittest.TestCase):
    def test_dataset_view_fallback_keeps_runtime_root_inside_code_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo_root = base

            source_images_dir = base / "TomatoMAP" / "TomatoMAP_seg" / "images"
            source_images_dir.mkdir(parents=True)
            (source_images_dir / "a.JPG").write_bytes(b"img")

            coco_dir = base / "code" / "outputs" / "paper1" / "baseline_v1" / "coco"
            coco_dir.mkdir(parents=True)
            for split in ("train", "val", "test"):
                with open(coco_dir / f"{split}.json", "w", encoding="utf-8") as f:
                    json.dump({"images": [], "annotations": [], "categories": []}, f)

            config = {
                "dataset": {
                    "source_images_dir": "TomatoMAP/TomatoMAP_seg/images",
                },
                "paths": {
                    "dataset_view_dir": "code/outputs/paper1/baseline_v1/dataset_view",
                    "dataset_view_images_dir": "code/outputs/paper1/baseline_v1/dataset_view/images",
                    "dataset_view_coco_dir": "code/outputs/paper1/baseline_v1/dataset_view/cocoOut",
                },
                "dataset_view": {
                    "enabled": True,
                    "link_images": True,
                    "allow_fallback_to_source_images": True,
                },
            }

            with mock.patch("src.experiments.paper1_baseline.os.symlink", side_effect=OSError("no symlink")), \
                 mock.patch("src.experiments.paper1_baseline.subprocess.run") as run_mock:
                run_mock.return_value = mock.Mock(returncode=1, stdout="", stderr="")
                result = _prepare_dataset_view(config, repo_root, coco_dir)

            dataset_view_dir = base / "code" / "outputs" / "paper1" / "baseline_v1" / "dataset_view"
            dataset_view_images_dir = dataset_view_dir / "images"
            dataset_view_coco_dir = dataset_view_dir / "cocoOut"

            self.assertEqual(result["data_dir"], str(dataset_view_dir))
            self.assertEqual(result["link_mode"], "copied_images")
            self.assertTrue((dataset_view_images_dir / "a.JPG").exists())
            self.assertTrue((dataset_view_coco_dir / "train.json").exists())
            self.assertTrue((dataset_view_coco_dir / "val.json").exists())
            self.assertTrue((dataset_view_coco_dir / "test.json").exists())


if __name__ == "__main__":
    unittest.main()


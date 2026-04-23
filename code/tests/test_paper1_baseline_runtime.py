import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from src.experiments.paper1_baseline import _prepare_dataset_view, freeze_split_once


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
                "experiment_id": "test_exp",
                "paper_track": "test_track",
                "class_names": ["tomato"],
                "image_size": {"width": 1024, "height": 1024},
                "training": {
                    "engine_entrypoint": "test.py",
                    "task": "seg",
                    "action": "train",
                    "model": "test.pt",
                    "epochs": 1,
                    "batch_size": 1,
                    "lr": 0.001,
                    "patience": 1,
                },
                "evaluation": {
                    "engine_entrypoint": "test.py",
                    "task": "seg",
                    "action": "eval",
                    "model_path": "test.pth",
                },
                "split": {
                    "manifest_path": "code/configs/paper1/split_manifest_v1.json",
                },
                "dataset": {
                    "source_images_dir": "TomatoMAP/TomatoMAP_seg/images",
                    "source_labels_dir": "TomatoMAP/TomatoMAP_seg/labels",
                    "categories_file": "TomatoMAP/TomatoMAP_seg/labels/isat.yaml",
                },
                "paths": {
                    "dataset_view_dir": "code/outputs/paper1/baseline_v1/dataset_view",
                    "dataset_view_images_dir": "code/outputs/paper1/baseline_v1/dataset_view/images",
                    "dataset_view_coco_dir": "code/outputs/paper1/baseline_v1/dataset_view/cocoOut",
                    "baseline_root": "code/outputs",
                    "coco_dir": "code/outputs/coco",
                    "train_output_dir": "code/outputs/train",
                    "logs_dir": "code/outputs/logs",
                },
                "dataset_view": {
                    "enabled": True,
                    "link_images": True,
                    "allow_fallback_to_source_images": True,
                },
                "artifacts": {
                    "split_summary": "code/outputs/paper1/baseline_v1/split_summary.json",
                    "run_manifest": "code/outputs/run.json",
                    "train_args": "code/outputs/args.json",
                    "eval_metrics": "code/outputs/metrics.json",
                    "artifact_index": "code/outputs/idx.json",
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

    def test_freeze_rebuilds_missing_coco_when_manifest_is_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)

            source_images_dir = base / "TomatoMAP" / "TomatoMAP_seg" / "images"
            source_labels_dir = base / "TomatoMAP" / "TomatoMAP_seg" / "labels"
            source_images_dir.mkdir(parents=True)
            source_labels_dir.mkdir(parents=True)

            splits = {
                "train": ["a.JPG"],
                "val": ["b.JPG"],
                "test": ["c.JPG"],
            }
            for file_name in ["a.JPG", "b.JPG", "c.JPG"]:
                (source_images_dir / file_name).write_bytes(b"img")
                with open(source_labels_dir / f"{Path(file_name).stem}.json", "w", encoding="utf-8") as f:
                    json.dump({"info": {"name": file_name}, "objects": []}, f)

            manifest_path = base / "code" / "configs" / "paper1" / "split_manifest_v1.json"
            manifest_path.parent.mkdir(parents=True)
            manifest = {"frozen": True, "splits": splits}
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f)

            config = {
                "experiment_id": "test_exp",
                "paper_track": "test_track",
                "class_names": ["tomato"],
                "image_size": {"width": 1024, "height": 1024},
                "training": {
                    "engine_entrypoint": "test.py",
                    "task": "seg",
                    "action": "train",
                    "model": "test.pt",
                    "epochs": 1,
                    "batch_size": 1,
                    "lr": 0.001,
                    "patience": 1,
                },
                "evaluation": {
                    "engine_entrypoint": "test.py",
                    "task": "seg",
                    "action": "eval",
                    "model_path": "test.pth",
                },
                "split": {
                    "manifest_path": "code/configs/paper1/split_manifest_v1.json",
                    "train_ratio": 0.7,
                    "val_ratio": 0.2,
                    "seed": 888,
                },
                "paths": {
                    "coco_dir": "code/outputs/paper1/baseline_v1/coco",
                    "dataset_view_dir": "code/outputs/paper1/baseline_v1/dataset_view",
                    "dataset_view_images_dir": "code/outputs/paper1/baseline_v1/dataset_view/images",
                    "dataset_view_coco_dir": "code/outputs/paper1/baseline_v1/dataset_view/cocoOut",
                    "baseline_root": "code/outputs",
                    "train_output_dir": "code/outputs/train",
                    "logs_dir": "code/outputs/logs",
                },
                "dataset": {
                    "source_images_dir": "TomatoMAP/TomatoMAP_seg/images",
                    "source_labels_dir": "TomatoMAP/TomatoMAP_seg/labels",
                    "categories_file": "TomatoMAP/TomatoMAP_seg/labels/isat.yaml",
                },
                "dataset_view": {
                    "enabled": False,
                    "link_images": True,
                    "allow_fallback_to_source_images": True,
                },
                "artifacts": {
                    "split_summary": "code/outputs/paper1/baseline_v1/split_summary.json",
                    "run_manifest": "code/outputs/run.json",
                    "train_args": "code/outputs/args.json",
                    "eval_metrics": "code/outputs/metrics.json",
                    "artifact_index": "code/outputs/idx.json",
                },
            }

            config_path = base / "code" / "configs" / "paper1" / "baseline_v1.kaggle.json"
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f)

            def fake_convert(**kwargs):
                out_dir = Path(kwargs["output_dir"])
                out_dir.mkdir(parents=True, exist_ok=True)
                for split, files in splits.items():
                    payload = {
                        "images": [{"file_name": n, "id": i + 1} for i, n in enumerate(files)],
                        "annotations": [],
                        "categories": [{"id": 1, "name": "tomato", "supercategory": "none"}],
                    }
                    with open(out_dir / f"{split}.json", "w", encoding="utf-8") as wf:
                        json.dump(payload, wf)

            with mock.patch("src.experiments.paper1_baseline.convert_isat_folder_to_coco", side_effect=fake_convert):
                result = freeze_split_once(config_path, base, force=False)

            self.assertEqual(result["status"], "recreated_coco_from_frozen_split")
            self.assertEqual(result["summary"]["counts"]["train"], 1)
            self.assertEqual(result["summary"]["counts"]["val"], 1)
            self.assertEqual(result["summary"]["counts"]["test"], 1)


if __name__ == "__main__":
    unittest.main()


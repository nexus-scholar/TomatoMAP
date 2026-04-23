#!/usr/bin/env python3
"""TomatoMAP Paper 1 entry point for orchestration and local seg engine mode."""

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).parent
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.experiments.paper1_baseline import run_stage
from src.experiments.config import ExperimentConfig


def load_config(config_path: Path) -> ExperimentConfig:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return ExperimentConfig.load(config_path)


def train(config_path: Path, repo_root: Optional[Path] = None) -> Dict[str, Any]:
    repo_root = repo_root or REPO_ROOT
    config = load_config(config_path)
    return run_stage(config_path, repo_root, "train")


def evaluate(config_path: Path, repo_root: Optional[Path] = None) -> Dict[str, Any]:
    repo_root = repo_root or REPO_ROOT
    config = load_config(config_path)
    return run_stage(config_path, repo_root, "eval")


def _copy_or_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def _build_yolo_runtime_dataset(data_dir: Path, output_dir: Path, num_classes: int) -> Path:
    coco_dir = data_dir / "cocoOut"
    images_root = data_dir / "images"
    if not coco_dir.exists() or not images_root.exists():
        raise FileNotFoundError(f"Expected dataset view with images/ and cocoOut/ under: {data_dir}")

    runtime_dir = output_dir / "_runtime_yolo"
    (runtime_dir / "images").mkdir(parents=True, exist_ok=True)
    (runtime_dir / "labels").mkdir(parents=True, exist_ok=True)

    # Build a stable category mapping once from COCO categories (same schema across splits).
    category_entries = []
    for split in ("train", "val", "test"):
        split_json = coco_dir / f"{split}.json"
        if not split_json.exists():
            continue
        payload = json.loads(split_json.read_text(encoding="utf-8"))
        raw_categories = payload.get("categories", [])
        if raw_categories:
            category_entries = sorted(raw_categories, key=lambda c: int(c.get("id", 0)))
            break

    if not category_entries:
        raise ValueError(f"No categories found in COCO split files under: {coco_dir}")

    if int(num_classes) <= 1:
        # Paper 1 single-class mode: collapse all source categories to class 0.
        id_to_idx = {int(c["id"]): 0 for c in category_entries}
        yolo_names = [str(category_entries[0].get("name", "class_0"))]
    else:
        # Keep original category semantics and names for multiclass runs.
        selected_categories = category_entries[: int(num_classes)]
        id_to_idx = {int(c["id"]): i for i, c in enumerate(selected_categories)}
        yolo_names = [str(c.get("name", f"class_{i}")) for i, c in enumerate(selected_categories)]

    for split in ("train", "val", "test"):
        split_json = coco_dir / f"{split}.json"
        if not split_json.exists():
            continue

        payload = json.loads(split_json.read_text(encoding="utf-8"))
        images = payload.get("images", [])
        annotations = payload.get("annotations", [])
        # category mapping is computed once above to keep consistent ids across splits.

        anns_by_img: Dict[int, list[dict]] = {}
        for ann in annotations:
            anns_by_img.setdefault(ann["image_id"], []).append(ann)

        for img in images:
            file_name = img["file_name"]
            width = float(img["width"])
            height = float(img["height"])
            image_id = img["id"]

            src_img = images_root / file_name
            dst_img = runtime_dir / "images" / split / file_name
            _copy_or_link(src_img, dst_img)

            lbl_path = (runtime_dir / "labels" / split / file_name).with_suffix(".txt")
            lbl_path.parent.mkdir(parents=True, exist_ok=True)

            lines: list[str] = []
            for ann in anns_by_img.get(image_id, []):
                cid = ann.get("category_id")
                if cid not in id_to_idx:
                    continue
                seg = ann.get("segmentation", [])
                if not seg:
                    continue

                if isinstance(seg, list) and seg and isinstance(seg[0], list):
                    poly = seg[0]
                elif isinstance(seg, list):
                    poly = seg
                else:
                    continue

                if len(poly) < 6 or len(poly) % 2 != 0:
                    continue

                norm = []
                for i in range(0, len(poly), 2):
                    x = max(0.0, min(1.0, float(poly[i]) / width))
                    y = max(0.0, min(1.0, float(poly[i + 1]) / height))
                    norm.extend([x, y])
                lines.append("{} {}".format(id_to_idx[int(cid)], " ".join(f"{v:.6f}" for v in norm)))

            lbl_path.write_text("\n".join(lines), encoding="utf-8")

    names = "\n".join([f'  {i}: "{name}"' for i, name in enumerate(yolo_names)])
    data_yaml = runtime_dir / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {runtime_dir.as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                names,
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return data_yaml


def _make_json_safe(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {str(k): _make_json_safe(v) for k, v in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_make_json_safe(v) for v in payload]
    if isinstance(payload, (str, int, float, bool)) or payload is None:
        return payload
    try:
        return float(payload)
    except Exception:
        return str(payload)


def _write_backend_metadata(output_dir: Path, backend: str, model: str) -> None:
    meta_path = output_dir / "backend.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps({"backend": backend, "model": model}, indent=2), encoding="utf-8")


def _read_backend_metadata(output_dir: Path) -> Optional[str]:
    meta_path = output_dir / "backend.json"
    if not meta_path.exists():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        backend = payload.get("backend")
        return str(backend) if backend else None
    except Exception:
        return None


def _seg_train(args: argparse.Namespace) -> Dict[str, Any]:
    model_name = str(args.model).lower()

    try:
        from ultralytics import YOLO
        import torch
    except Exception as exc:
        raise RuntimeError("ultralytics is required for YOLO training.") from exc

    cuda_available = torch.cuda.is_available()
    cuda_count = torch.cuda.device_count() if cuda_available else 0
    if cuda_available and cuda_count > 0:
        print(f"[device] CUDA available. count={cuda_count}, active={torch.cuda.get_device_name(0)}")
    else:
        print("[device] CUDA not available. Training will run on CPU unless device is configured externally.")

    data_yaml = _build_yolo_runtime_dataset(Path(args.data_dir), Path(args.output_dir), int(args.num_classes))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(data_yaml),
        "epochs": int(args.epochs),
        "batch": int(args.batch_size),
        "lr0": float(args.lr),
        "patience": int(args.patience),
        "project": str(Path(args.output_dir)),
        "name": "yolo_seg",
        "exist_ok": True,
    }
    if getattr(args, "device", None) not in (None, ""):
        train_kwargs["device"] = str(args.device)

    run = model.train(
        **train_kwargs,
    )

    best = Path(run.save_dir) / "weights" / "best.pt"
    if best.exists():
        shutil.copy2(best, Path(args.output_dir) / "model_best.pth")

    _write_backend_metadata(Path(args.output_dir), "ultralytics", str(args.model))

    return {"status": "ok", "backend": "ultralytics", "save_dir": str(run.save_dir)}


def _seg_eval(args: argparse.Namespace) -> Dict[str, Any]:
    model_name = str(args.model_path).lower()
    output_dir = Path(args.output_dir)

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("ultralytics is required for YOLO evaluation.") from exc

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = Path(args.output_dir) / model_path

    if not model_path.exists() and model_path.suffix == ".pth":
        alt = model_path.with_suffix(".pt")
        if alt.exists():
            model_path = alt

    data_yaml = _build_yolo_runtime_dataset(Path(args.data_dir), Path(args.output_dir), 1)
    model = YOLO(str(model_path))
    results = model.val(data=str(data_yaml), split="test")

    result_dict = getattr(results, "results_dict", {}) or {}
    safe_payload = {}
    for k, v in result_dict.items():
        try:
            safe_payload[k] = float(v)
        except Exception:
            safe_payload[k] = str(v)

    out = output_dir / f"test_results_{Path(args.model_path).stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(safe_payload, indent=2), encoding="utf-8")
    return {"status": "ok", "backend": "ultralytics", "metrics_file": str(out)}


def main() -> int:
    parser = argparse.ArgumentParser(description="TomatoMAP Paper 1 Baseline Segmentation Training")
    subparsers = parser.add_subparsers(dest="command", help="Command to run", required=True)

    train_parser = subparsers.add_parser("train", help="Orchestrated pipeline train stage")
    train_parser.add_argument("--config", type=Path, required=True)
    train_parser.set_defaults(func=lambda args: train(args.config))

    eval_parser = subparsers.add_parser("eval", help="Orchestrated pipeline eval stage")
    eval_parser.add_argument("--config", type=Path, required=True)
    eval_parser.set_defaults(func=lambda args: evaluate(args.config))

    # Engine compatibility mode used by run_stage subprocess commands.
    seg_parser = subparsers.add_parser("seg", help="Internal seg engine command mode")
    seg_sub = seg_parser.add_subparsers(dest="action", required=True)

    seg_train = seg_sub.add_parser("train", help="Train seg backend")
    seg_train.add_argument("--data-dir", required=True)
    seg_train.add_argument("--model", required=True)
    seg_train.add_argument("--epochs", type=int, required=True)
    seg_train.add_argument("--lr", type=float, required=True)
    seg_train.add_argument("--batch-size", type=int, required=True)
    seg_train.add_argument("--patience", type=int, required=True)
    seg_train.add_argument("--output-dir", required=True)
    seg_train.add_argument("--num-classes", type=int, default=1)
    seg_train.add_argument("--device", default="", help="Optional device override for YOLO (e.g., '0' or 'cpu').")
    seg_train.set_defaults(func=_seg_train)

    seg_eval = seg_sub.add_parser("eval", help="Evaluate seg backend")
    seg_eval.add_argument("--data-dir", required=True)
    seg_eval.add_argument("--output-dir", required=True)
    seg_eval.add_argument("--model-path", required=True)
    seg_eval.set_defaults(func=_seg_eval)

    args = parser.parse_args()

    try:
        result = args.func(args)
        print(json.dumps(result, indent=2))
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())


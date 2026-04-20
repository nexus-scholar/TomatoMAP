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


def load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_config(config: Dict[str, Any]) -> None:
    required_keys = ["experiment_id", "dataset", "split", "paths", "training", "evaluation"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")


def train(config_path: Path, repo_root: Optional[Path] = None) -> Dict[str, Any]:
    repo_root = repo_root or REPO_ROOT
    config = load_config(config_path)
    validate_config(config)
    return run_stage(config_path, repo_root, "train")


def evaluate(config_path: Path, repo_root: Optional[Path] = None) -> Dict[str, Any]:
    repo_root = repo_root or REPO_ROOT
    config = load_config(config_path)
    validate_config(config)
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

    for split in ("train", "val", "test"):
        split_json = coco_dir / f"{split}.json"
        if not split_json.exists():
            continue

        payload = json.loads(split_json.read_text(encoding="utf-8"))
        images = payload.get("images", [])
        annotations = payload.get("annotations", [])
        categories = payload.get("categories", [])

        cat_ids = sorted([c["id"] for c in categories])
        if int(num_classes) <= 1:
            # Paper 1 single-class mode: collapse all source categories to class 0.
            id_to_idx = {cid: 0 for cid in cat_ids}
        else:
            # Guard against out-of-range labels by mapping only the first num_classes ids.
            valid_ids = cat_ids[: int(num_classes)]
            id_to_idx = {cid: i for i, cid in enumerate(valid_ids)}

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
                lines.append("{} {}".format(id_to_idx[cid], " ".join(f"{v:.6f}" for v in norm)))

            lbl_path.write_text("\n".join(lines), encoding="utf-8")

    names = "\n".join([f"  {i}: class_{i}" for i in range(max(1, num_classes))])
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


def _register_detectron2_split(name: str, images_root: Path, coco_json: Path) -> None:
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data.datasets import register_coco_instances

    if name in DatasetCatalog.list():
        DatasetCatalog.remove(name)
    register_coco_instances(name, {}, str(coco_json), str(images_root))
    MetadataCatalog.get(name)


def _build_detectron2_cfg(args: argparse.Namespace, train_name: str, val_name: str):
    from detectron2 import model_zoo
    from detectron2.config import get_cfg

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.model))
    cfg.DATASETS.TRAIN = (train_name,)
    cfg.DATASETS.TEST = (val_name,)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = int(args.batch_size)
    cfg.SOLVER.BASE_LR = float(args.lr)
    cfg.SOLVER.STEPS = []
    cfg.SOLVER.GAMMA = 1.0
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = int(args.num_classes)
    cfg.OUTPUT_DIR = str(Path(args.output_dir))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.model)
    return cfg


def _seg_train_detectron2(args: argparse.Namespace) -> Dict[str, Any]:
    try:
        from detectron2.engine import DefaultTrainer
        from detectron2.data import DatasetCatalog
    except Exception as exc:
        raise RuntimeError("detectron2 is required for Detectron2 training in the new engine.") from exc

    data_dir = Path(args.data_dir)
    coco_dir = data_dir / "cocoOut"
    images_root = data_dir / "images"
    train_json = coco_dir / "train.json"
    val_json = coco_dir / "val.json"
    if not train_json.exists() or not val_json.exists() or not images_root.exists():
        raise FileNotFoundError(f"Expected dataset view with images/ and cocoOut/train.json,val.json under: {data_dir}")

    run_tag = str(abs(hash(Path(args.output_dir).as_posix())))
    train_name = f"tomatomap_train_{run_tag}"
    val_name = f"tomatomap_val_{run_tag}"
    _register_detectron2_split(train_name, images_root, train_json)
    _register_detectron2_split(val_name, images_root, val_json)

    cfg = _build_detectron2_cfg(args, train_name, val_name)
    train_items = DatasetCatalog.get(train_name)
    iters_per_epoch = max(1, math.ceil(len(train_items) / max(1, int(args.batch_size))))
    cfg.SOLVER.MAX_ITER = int(args.epochs) * iters_per_epoch

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    final_model = Path(args.output_dir) / "model_final.pth"
    if final_model.exists():
        shutil.copy2(final_model, Path(args.output_dir) / "model_best.pth")

    _write_backend_metadata(Path(args.output_dir), "detectron2", str(args.model))
    return {"status": "ok", "backend": "detectron2", "output_dir": str(args.output_dir)}


def _seg_eval_detectron2(args: argparse.Namespace) -> Dict[str, Any]:
    try:
        from detectron2 import model_zoo
        from detectron2.checkpoint import DetectionCheckpointer
        from detectron2.config import get_cfg
        from detectron2.data import build_detection_test_loader
        from detectron2.engine import DefaultPredictor
        from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    except Exception as exc:
        raise RuntimeError("detectron2 is required for Detectron2 evaluation in the new engine.") from exc

    data_dir = Path(args.data_dir)
    coco_dir = data_dir / "cocoOut"
    images_root = data_dir / "images"
    test_json = coco_dir / "test.json"
    if not test_json.exists() or not images_root.exists():
        raise FileNotFoundError(f"Expected dataset view with images/ and cocoOut/test.json under: {data_dir}")

    run_tag = str(abs(hash(Path(args.output_dir).as_posix())))
    test_name = f"tomatomap_test_{run_tag}"
    _register_detectron2_split(test_name, images_root, test_json)

    backend_meta = _read_backend_metadata(Path(args.output_dir))
    model_hint = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
    if backend_meta == "detectron2":
        meta_path = Path(args.output_dir) / "backend.json"
        try:
            model_hint = json.loads(meta_path.read_text(encoding="utf-8")).get("model", model_hint)
        except Exception:
            pass

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_hint))
    cfg.DATASETS.TEST = (test_name,)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.OUTPUT_DIR = str(Path(args.output_dir))

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = Path(args.output_dir) / model_path
    if not model_path.exists():
        fallback = Path(args.output_dir) / "model_final.pth"
        if fallback.exists():
            model_path = fallback
    if not model_path.exists():
        raise FileNotFoundError(f"Detectron2 model checkpoint not found: {model_path}")

    cfg.MODEL.WEIGHTS = str(model_path)
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator(test_name, cfg, False, output_dir=str(Path(args.output_dir) / "eval"))
    val_loader = build_detection_test_loader(cfg, test_name)
    results = inference_on_dataset(predictor.model, val_loader, evaluator)

    safe_payload = _make_json_safe(results)
    out = Path(args.output_dir) / f"test_results_{Path(args.model_path).stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(safe_payload, indent=2), encoding="utf-8")
    return {"status": "ok", "backend": "detectron2", "metrics_file": str(out)}


def _seg_train(args: argparse.Namespace) -> Dict[str, Any]:
    model_name = str(args.model).lower()
    if "yolo" not in model_name and "coco-instancesegmentation" not in model_name:
        raise RuntimeError("Unsupported segmentation model string for new engine.")

    if "coco-instancesegmentation" in model_name:
        return _seg_train_detectron2(args)

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("ultralytics is required for YOLO training in the new engine.") from exc

    data_yaml = _build_yolo_runtime_dataset(Path(args.data_dir), Path(args.output_dir), int(args.num_classes))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    run = model.train(
        data=str(data_yaml),
        epochs=int(args.epochs),
        batch=int(args.batch_size),
        lr0=float(args.lr),
        patience=int(args.patience),
        project=str(Path(args.output_dir)),
        name="yolo_seg",
        exist_ok=True,
    )

    best = Path(run.save_dir) / "weights" / "best.pt"
    if best.exists():
        shutil.copy2(best, Path(args.output_dir) / "model_best.pth")

    _write_backend_metadata(Path(args.output_dir), "ultralytics", str(args.model))

    return {"status": "ok", "backend": "ultralytics", "save_dir": str(run.save_dir)}


def _seg_eval(args: argparse.Namespace) -> Dict[str, Any]:
    model_name = str(args.model_path).lower()
    output_dir = Path(args.output_dir)
    backend = _read_backend_metadata(output_dir)
    if backend == "detectron2":
        return _seg_eval_detectron2(args)

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("ultralytics is required for YOLO evaluation in the new engine.") from exc

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = Path(args.output_dir) / model_path

    if not model_path.exists() and model_path.suffix == ".pth":
        alt = model_path.with_suffix(".pt")
        if alt.exists():
            model_path = alt

    if "yolo" not in model_name and model_path.suffix not in {".pt", ".pth"}:
        raise RuntimeError("New engine currently supports YOLO segmentation evaluation only.")

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


import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from ..data.isat_to_coco import convert_isat_folder_to_coco
from ..utils.io import write_json
from ..utils.paths import ensure_dir
from .split_validation import load_json, validate_manifest_against_files


def resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (repo_root / path)


def load_config(config_path: Path) -> dict:
    return load_json(config_path)


def _build_manifest(config: dict, split_lists: dict[str, list[str]]) -> dict:
    return {
        "manifest_version": "v1",
        "frozen": True,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "seed": config["split"]["seed"],
        "train_ratio": config["split"]["train_ratio"],
        "val_ratio": config["split"]["val_ratio"],
        "categories_file": config["dataset"]["categories_file"],
        "class_names": config.get("class_names", []),
        "splits": split_lists,
    }


def _read_split_lists(coco_dir: Path) -> dict[str, list[str]]:
    split_lists: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        payload = load_json(coco_dir / f"{split}.json")
        split_lists[split] = sorted([img["file_name"] for img in payload.get("images", [])])
    return split_lists


def _populate_dataset_view_images(source_images_dir: Path, dataset_view_images_dir: Path) -> str:
    if dataset_view_images_dir.exists():
        return "existing"

    dataset_view_images_dir.parent.mkdir(parents=True, exist_ok=True)

    try:
        os.symlink(source_images_dir, dataset_view_images_dir, target_is_directory=True)
        return "symlink"
    except OSError:
        pass

    if os.name == "nt":
        # Windows fallback: directory junction for non-admin environments.
        cmd = ["cmd", "/c", "mklink", "/J", str(dataset_view_images_dir), str(source_images_dir)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return "junction"

    shutil.copytree(source_images_dir, dataset_view_images_dir, dirs_exist_ok=True)
    return "copied_images"


def _prepare_dataset_view(config: dict, repo_root: Path, coco_dir: Path) -> dict:
    dataset_view_cfg = config.get("dataset_view", {})
    dataset_view_dir = resolve_path(repo_root, config["paths"]["dataset_view_dir"])
    dataset_view_images_dir = resolve_path(repo_root, config["paths"]["dataset_view_images_dir"])
    dataset_view_coco_dir = resolve_path(repo_root, config["paths"]["dataset_view_coco_dir"])

    if not dataset_view_cfg.get("enabled", True):
        return {"enabled": False, "data_dir": str(dataset_view_dir), "link_mode": "disabled"}

    ensure_dir(dataset_view_dir)
    ensure_dir(dataset_view_coco_dir)

    for split in ("train", "val", "test"):
        shutil.copy2(coco_dir / f"{split}.json", dataset_view_coco_dir / f"{split}.json")

    link_mode = "disabled"
    if dataset_view_cfg.get("link_images", True):
        source_images_dir = resolve_path(repo_root, config["dataset"]["source_images_dir"])
        link_mode = _populate_dataset_view_images(source_images_dir, dataset_view_images_dir)

    if link_mode in {"symlink", "junction", "existing", "copied_images"}:
        return {"enabled": True, "data_dir": str(dataset_view_dir), "link_mode": link_mode}

    if dataset_view_cfg.get("allow_fallback_to_source_images", True):
        source_images_dir = resolve_path(repo_root, config["dataset"]["source_images_dir"])
        link_mode = _populate_dataset_view_images(source_images_dir, dataset_view_images_dir)
        return {"enabled": True, "data_dir": str(dataset_view_dir), "link_mode": link_mode}

    raise RuntimeError("Dataset view image link/junction creation failed and fallback is disabled.")


def freeze_split_once(config_path: Path, repo_root: Path, force: bool = False) -> dict:
    config = load_config(config_path)

    manifest_path = resolve_path(repo_root, config["split"]["manifest_path"])
    coco_dir = resolve_path(repo_root, config["paths"]["coco_dir"])
    source_images_dir = resolve_path(repo_root, config["dataset"]["source_images_dir"])
    source_labels_dir = resolve_path(repo_root, config["dataset"]["source_labels_dir"])
    categories_file = resolve_path(repo_root, config["dataset"]["categories_file"])

    if manifest_path.exists() and not force:
        manifest = load_json(manifest_path)
        if manifest.get("frozen", False):
            try:
                summary = validate_manifest_against_files(manifest, coco_dir, source_images_dir, source_labels_dir)
                return {"status": "reused_existing_frozen_split", "summary": summary}
            except FileNotFoundError:
                # Frozen manifest is present, but COCO splits are missing (common on fresh Kaggle clone).
                ensure_dir(coco_dir)
                convert_isat_folder_to_coco(
                    task_dir=str(source_images_dir),
                    label_dir=str(source_labels_dir),
                    categories_path=str(categories_file),
                    output_dir=str(coco_dir),
                    train_ratio=float(config["split"]["train_ratio"]),
                    val_ratio=float(config["split"]["val_ratio"]),
                    seed=int(config["split"]["seed"]),
                )
                summary = validate_manifest_against_files(manifest, coco_dir, source_images_dir, source_labels_dir)

                split_summary_path = resolve_path(repo_root, config["artifacts"]["split_summary"])
                write_json(split_summary_path, summary)

                dataset_view_status = _prepare_dataset_view(config, repo_root, coco_dir)
                return {
                    "status": "recovered_existing_frozen_split",
                    "manifest_path": str(manifest_path),
                    "summary": summary,
                    "dataset_view": dataset_view_status,
                }

    ensure_dir(coco_dir)

    convert_isat_folder_to_coco(
        task_dir=str(source_images_dir),
        label_dir=str(source_labels_dir),
        categories_path=str(categories_file),
        output_dir=str(coco_dir),
        train_ratio=float(config["split"]["train_ratio"]),
        val_ratio=float(config["split"]["val_ratio"]),
        seed=int(config["split"]["seed"]),
    )

    split_lists = _read_split_lists(coco_dir)
    manifest = _build_manifest(config, split_lists)
    write_json(manifest_path, manifest)

    summary = validate_manifest_against_files(manifest, coco_dir, source_images_dir, source_labels_dir)

    split_summary_path = resolve_path(repo_root, config["artifacts"]["split_summary"])
    write_json(split_summary_path, summary)

    dataset_view_status = _prepare_dataset_view(config, repo_root, coco_dir)

    return {
        "status": "frozen",
        "manifest_path": str(manifest_path),
        "summary": summary,
        "dataset_view": dataset_view_status,
    }


def _run_subprocess(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(command, capture_output=True, text=True)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("COMMAND:\n")
        f.write(" ".join(command) + "\n\n")
        f.write("STDOUT:\n")
        f.write(result.stdout)
        f.write("\nSTDERR:\n")
        f.write(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"Subprocess failed ({result.returncode}). See log: {log_path}")


def _resolve_data_dir(config: dict, repo_root: Path, coco_dir: Path) -> str:
    dataset_view_status = _prepare_dataset_view(config, repo_root, coco_dir)
    return dataset_view_status["data_dir"]


def _write_artifact_index(config: dict, repo_root: Path) -> None:
    artifact_index_path = resolve_path(repo_root, config["artifacts"]["artifact_index"])
    payload = {
        "run_manifest": config["artifacts"]["run_manifest"],
        "split_summary": config["artifacts"]["split_summary"],
        "train_args": config["artifacts"]["train_args"],
        "eval_metrics": config["artifacts"]["eval_metrics"],
    }
    write_json(artifact_index_path, payload)


def run_stage(config_path: Path, repo_root: Path, stage: str) -> dict:
    config = load_config(config_path)
    manifest_path = resolve_path(repo_root, config["split"]["manifest_path"])
    manifest = load_json(manifest_path)

    coco_dir = resolve_path(repo_root, config["paths"]["coco_dir"])
    source_images_dir = resolve_path(repo_root, config["dataset"]["source_images_dir"])
    source_labels_dir = resolve_path(repo_root, config["dataset"]["source_labels_dir"])
    validate_manifest_against_files(manifest, coco_dir, source_images_dir, source_labels_dir)

    data_dir = _resolve_data_dir(config, repo_root, coco_dir)

    train_output_dir = resolve_path(repo_root, config["paths"]["train_output_dir"])
    logs_dir = resolve_path(repo_root, config["paths"]["logs_dir"])
    ensure_dir(train_output_dir)
    ensure_dir(logs_dir)

    python_exe = "python"
    if stage == "train":
        spec = config["training"]
        engine = resolve_path(repo_root, spec["engine_entrypoint"])
        num_classes = len(config.get("class_names", []))

        command = [
            python_exe,
            str(engine),
            spec["task"],
            spec["action"],
            "--data-dir",
            str(data_dir),
            "--model",
            spec["model"],
            "--epochs",
            str(spec["epochs"]),
            "--lr",
            str(spec["lr"]),
            "--batch-size",
            str(spec["batch_size"]),
            "--patience",
            str(spec["patience"]),
            "--output-dir",
            str(train_output_dir),
            "--num-classes",
            str(num_classes),
        ]

        log_path = logs_dir / "train.log"
        _run_subprocess(command, log_path)

        train_args_path = resolve_path(repo_root, config["artifacts"]["train_args"])
        write_json(
            train_args_path,
            {
                "command": command,
                "data_dir": data_dir,
                "image_size": config.get("image_size", {}),
                "class_names": config.get("class_names", []),
            },
        )

    elif stage == "eval":
        spec = config["evaluation"]
        engine = resolve_path(repo_root, spec["engine_entrypoint"])
        command = [
            python_exe,
            str(engine),
            spec["task"],
            spec["action"],
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(train_output_dir),
            "--model-path",
            spec["model_path"],
        ]

        log_path = logs_dir / "eval.log"
        _run_subprocess(command, log_path)

        # Prefer the expected evaluation artifact from the legacy runner, if available.
        expected_result = train_output_dir / f"test_results_{Path(spec['model_path']).stem}.json"
        fallback_result = train_output_dir / "test_results_model_final.json"

        eval_metrics_path = resolve_path(repo_root, config["artifacts"]["eval_metrics"])
        if expected_result.exists():
            write_json(eval_metrics_path, load_json(expected_result))
        elif fallback_result.exists():
            write_json(eval_metrics_path, load_json(fallback_result))
        else:
            write_json(
                eval_metrics_path,
                {
                    "note": "No legacy evaluation json artifact found.",
                    "expected": str(expected_result),
                    "fallback": str(fallback_result),
                },
            )

    else:
        raise ValueError("Stage must be either 'train' or 'eval'.")

    run_manifest_path = resolve_path(repo_root, config["artifacts"]["run_manifest"])
    write_json(
        run_manifest_path,
        {
            "stage": stage,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "config_path": str(config_path),
            "manifest_path": str(manifest_path),
            "data_dir": str(data_dir),
        },
    )

    _write_artifact_index(config, repo_root)

    return {"stage": stage, "status": "ok", "data_dir": data_dir}

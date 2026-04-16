import json
from pathlib import Path


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _split_file_sets(manifest: dict) -> dict[str, set[str]]:
    splits = manifest.get("splits", {})
    return {
        "train": set(splits.get("train", [])),
        "val": set(splits.get("val", [])),
        "test": set(splits.get("test", [])),
    }


def validate_manifest_structure(manifest: dict) -> None:
    if not manifest.get("frozen", False):
        raise ValueError("Split manifest is not frozen. Freeze it once before running baseline.")

    split_sets = _split_file_sets(manifest)
    overlap = (split_sets["train"] & split_sets["val"]) | (split_sets["train"] & split_sets["test"]) | (split_sets["val"] & split_sets["test"])
    if overlap:
        sample = sorted(list(overlap))[:5]
        raise ValueError(f"Split manifest has overlapping files across splits: {sample}")



def read_coco_image_lists(coco_dir: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        split_path = coco_dir / f"{split}.json"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing COCO split file: {split_path}")
        payload = load_json(split_path)
        result[split] = sorted([img["file_name"] for img in payload.get("images", [])])
    return result


def validate_manifest_against_files(
    manifest: dict,
    coco_dir: Path,
    images_dir: Path,
    labels_dir: Path,
) -> dict:
    validate_manifest_structure(manifest)

    coco_lists = read_coco_image_lists(coco_dir)
    manifest_lists = {
        "train": sorted(manifest.get("splits", {}).get("train", [])),
        "val": sorted(manifest.get("splits", {}).get("val", [])),
        "test": sorted(manifest.get("splits", {}).get("test", [])),
    }

    for split in ("train", "val", "test"):
        if manifest_lists[split] != coco_lists[split]:
            raise ValueError(f"Split mismatch for {split}: manifest and COCO lists differ.")

    missing_images: list[str] = []
    missing_labels: list[str] = []
    for split in ("train", "val", "test"):
        for file_name in manifest_lists[split]:
            image_path = images_dir / file_name
            if not image_path.exists():
                missing_images.append(str(image_path))
            label_path = labels_dir / f"{Path(file_name).stem}.json"
            if not label_path.exists():
                missing_labels.append(str(label_path))

    if missing_images:
        raise FileNotFoundError(f"Missing images referenced by manifest: {missing_images[:5]}")
    if missing_labels:
        raise FileNotFoundError(f"Missing labels referenced by manifest: {missing_labels[:5]}")

    return {
        "frozen": True,
        "counts": {split: len(manifest_lists[split]) for split in ("train", "val", "test")},
        "total": sum(len(manifest_lists[split]) for split in ("train", "val", "test")),
    }


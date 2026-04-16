import json
from pathlib import Path

from ..utils.io import write_lines
from ..utils.paths import validate_seg_dirs


def _collect_image_stems(img_dir: Path) -> set[str]:
    stems: set[str] = set()
    for pattern in ("*.JPG", "*.jpg", "*.JPEG", "*.jpeg", "*.png", "*.PNG"):
        stems.update(p.stem for p in img_dir.glob(pattern))
    return stems


def _collect_label_stems(lbl_dir: Path) -> set[str]:
    return {p.stem for p in lbl_dir.glob("*.json")}


def _collect_json_references(lbl_dir: Path) -> set[str]:
    refs: set[str] = set()
    for lbl_path in lbl_dir.glob("*.json"):
        try:
            with open(lbl_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                info = payload.get("info", {})
                name = info.get("name")
                if isinstance(name, str) and name.strip():
                    refs.add(Path(name).stem)
        except Exception:
            continue
    return refs


def audit_segmentation(img_dir: str, lbl_dir: str, report_path: str | None = None, write_report: bool = False) -> dict:
    status = validate_seg_dirs(img_dir, lbl_dir)
    if not status["ok"]:
        raise FileNotFoundError(f"Segmentation paths missing: {status}")

    img_path = Path(img_dir)
    lbl_path = Path(lbl_dir)

    img_stems = _collect_image_stems(img_path)
    lbl_stems = _collect_label_stems(lbl_path)
    json_refs = _collect_json_references(lbl_path)

    missing = sorted(img_stems - lbl_stems)

    if write_report and report_path:
        write_lines(report_path, [f"{name}.JPG" for name in missing])

    return {
        "total_images": len(img_stems),
        "total_labels": len(lbl_stems),
        "matched_images": len(img_stems.intersection(lbl_stems)),
        "missing_count": len(missing),
        "missing_labels": missing,
        "json_referenced_images": len(json_refs),
    }

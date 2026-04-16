from pathlib import Path


def as_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def ensure_dir(path: str | Path) -> Path:
    p = as_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def validate_seg_dirs(img_dir: str | Path, lbl_dir: str | Path) -> dict:
    img_path = as_path(img_dir)
    lbl_path = as_path(lbl_dir)
    return {
        "img_dir": str(img_path),
        "lbl_dir": str(lbl_path),
        "img_exists": img_path.exists(),
        "lbl_exists": lbl_path.exists(),
        "ok": img_path.exists() and lbl_path.exists(),
    }


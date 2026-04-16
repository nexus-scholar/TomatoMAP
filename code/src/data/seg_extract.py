import zipfile
from pathlib import Path

from ..utils.paths import ensure_dir


def extract_segmentation_labels(zip_path: str, dest_dir: str, dry_run: bool = False) -> dict:
    src = Path(zip_path)
    out_dir = ensure_dir(dest_dir)

    if not src.exists():
        raise FileNotFoundError(f"Zip file not found: {src}")

    extracted = 0
    planned: list[str] = []

    with zipfile.ZipFile(src, "r") as zf:
        json_members = [name for name in zf.namelist() if name.lower().endswith(".json")]

        for member in json_members:
            target = out_dir / Path(member).name
            planned.append(str(target))
            if dry_run:
                continue
            with zf.open(member) as rf, open(target, "wb") as wf:
                wf.write(rf.read())
            extracted += 1

    return {
        "zip_path": str(src),
        "dest_dir": str(out_dir),
        "json_in_zip": len(planned),
        "extracted": extracted,
        "planned_targets": planned,
        "dry_run": dry_run,
    }

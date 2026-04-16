import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.seg_extract import extract_segmentation_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract segmentation json labels from zip.")
    parser.add_argument("--zip-path", required=True, help="Path to zip file")
    parser.add_argument("--dest-dir", required=True, help="Destination directory")
    parser.add_argument("--dry-run", action="store_true", help="Only preview extraction targets")
    args = parser.parse_args()

    result = extract_segmentation_labels(
        zip_path=args.zip_path,
        dest_dir=args.dest_dir,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

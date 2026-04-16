import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.seg_audit import audit_segmentation


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit segmentation image/label matching.")
    parser.add_argument("--img-dir", required=True, help="Image directory")
    parser.add_argument("--lbl-dir", required=True, help="Label directory")
    parser.add_argument("--report-path", default=None, help="Optional missing-label report path")
    parser.add_argument("--write-report", action="store_true", help="Write missing report if set")
    args = parser.parse_args()

    result = audit_segmentation(
        img_dir=args.img_dir,
        lbl_dir=args.lbl_dir,
        report_path=args.report_path,
        write_report=args.write_report,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

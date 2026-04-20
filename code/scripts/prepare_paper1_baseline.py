import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.experiments.paper1_baseline import freeze_split_once


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze canonical Paper 1 baseline split once.")
    parser.add_argument("--config", required=True, help="Path to baseline_v1.json")
    parser.add_argument("--freeze-split", action="store_true", help="Required safety flag for split freezing")
    parser.add_argument("--force", action="store_true", help="Regenerate split/coco and overwrite manifest")
    args = parser.parse_args()

    if not args.freeze_split:
        raise ValueError("Use --freeze-split to explicitly acknowledge split freezing.")

    result = freeze_split_once(Path(args.config), REPO_ROOT, force=args.force)
    print(f"[prepare] status: {result.get('status')}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


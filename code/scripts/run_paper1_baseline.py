import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.experiments.paper1_baseline import run_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Paper 1 baseline train/eval using frozen split.")
    parser.add_argument("--config", required=True, help="Path to baseline_v1.json")
    parser.add_argument("--stage", required=True, choices=["train", "eval"], help="Pipeline stage")
    args = parser.parse_args()

    result = run_stage(Path(args.config), REPO_ROOT, args.stage)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


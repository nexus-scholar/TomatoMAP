#!/usr/bin/env python3
"""
TomatoMAP Paper 1 Training Entry Point (New greenfield implementation)

This module provides a clean entry point for training and evaluation workflows
using frozen splits and reproducible configurations.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Setup paths
ROOT = Path(__file__).parent
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.experiments.paper1_baseline import run_stage


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load experiment configuration from JSON file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return json.load(f)


def validate_config(config: Dict[str, Any]) -> None:
    """Validate required configuration fields."""
    required_keys = ["experiment_id", "dataset", "split", "paths", "training", "evaluation"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")


def train(config_path: Path, repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """
    Train a segmentation model using frozen split.

    Args:
        config_path: Path to baseline_v1.json config file.
        repo_root: Repository root path (auto-detected if None).

    Returns:
        Dictionary with training results.
    """
    repo_root = repo_root or REPO_ROOT
    config = load_config(config_path)
    validate_config(config)

    return run_stage(config_path, repo_root, "train")


def evaluate(config_path: Path, repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """
    Evaluate a trained segmentation model.

    Args:
        config_path: Path to baseline_v1.json config file.
        repo_root: Repository root path (auto-detected if None).

    Returns:
        Dictionary with evaluation metrics.
    """
    repo_root = repo_root or REPO_ROOT
    config = load_config(config_path)
    validate_config(config)

    return run_stage(config_path, repo_root, "eval")


def main():
    """CLI entry point for training and evaluation."""
    parser = argparse.ArgumentParser(
        description="TomatoMAP Paper 1 Baseline Segmentation Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train using YOLOv8
  python code/main.py train --config code/configs/paper1/baseline_v1.json
  
  # Evaluate a trained model
  python code/main.py eval --config code/configs/paper1/baseline_v1.json
  
  # Train YOLOv11 variant
  python code/main.py train --config code/configs/paper1/baseline_v1_yolo11.json
  
  # Train Detectron2 variant
  python code/main.py train --config code/configs/paper1/baseline_v1_detectron2.json
        """
    )

    # Main subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run", required=True)

    # Train command
    train_parser = subparsers.add_parser("train", help="Train segmentation model")
    train_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to configuration JSON file (e.g., code/configs/paper1/baseline_v1.json)"
    )
    train_parser.set_defaults(func=lambda args: train(args.config))

    # Evaluate command
    eval_parser = subparsers.add_parser("eval", help="Evaluate segmentation model")
    eval_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to configuration JSON file (e.g., code/configs/paper1/baseline_v1.json)"
    )
    eval_parser.set_defaults(func=lambda args: evaluate(args.config))

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


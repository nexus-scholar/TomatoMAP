from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator
from typing import Optional, List, Dict, Any
from pathlib import Path

class DatasetConfig(BaseModel):
    base_data_dir: Optional[str] = None
    source_images_dir: str
    source_labels_dir: str
    categories_file: str
    verified_labels_only: bool = True

class SplitConfig(BaseModel):
    manifest_path: str
    freeze_once: bool = True
    regenerate_on_run: bool = False
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    seed: int = 888

class ImageSizeConfig(BaseModel):
    width: int = 1024
    height: int = 1024

class PathsConfig(BaseModel):
    base_output_dir: Optional[str] = None
    baseline_root: str
    coco_dir: str
    dataset_view_dir: str
    dataset_view_images_dir: str
    dataset_view_coco_dir: str
    train_output_dir: str
    logs_dir: str

class DatasetViewConfig(BaseModel):
    enabled: bool = True
    link_images: bool = True
    allow_fallback_to_source_images: bool = True

class TrainingConfig(BaseModel):
    engine_entrypoint: str = "code/main.py"
    task: str = "seg"
    action: str = "train"
    model: str
    epochs: int
    batch_size: int
    lr: float
    patience: int
    device: Optional[str] = None

class EvaluationConfig(BaseModel):
    engine_entrypoint: str = "code/main.py"
    task: str = "seg"
    action: str = "eval"
    model_path: str

class ArtifactsConfig(BaseModel):
    run_manifest: str
    split_summary: str
    train_args: str
    eval_metrics: str
    artifact_index: str

class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    experiment_id: str
    paper_track: str
    seed: int = 888
    dataset: DatasetConfig
    split: SplitConfig
    class_names: List[str]
    image_size: ImageSizeConfig
    paths: PathsConfig
    dataset_view: DatasetViewConfig = DatasetViewConfig()
    training: TrainingConfig
    evaluation: EvaluationConfig
    artifacts: ArtifactsConfig

    @model_validator(mode="before")
    @classmethod
    def expand_paths(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        exp_id = data.get("experiment_id", "exp_default")
        track = data.get("paper_track", "paper1")

        # 1) Auto map dataset logic
        ds = data.get("dataset", {})
        if isinstance(ds, dict):
            base_data = ds.get("base_data_dir")
            if base_data:
                ds.setdefault("source_images_dir", f"{base_data}/images")
                ds.setdefault("source_labels_dir", f"{base_data}/labels")
                ds.setdefault("categories_file", f"{base_data}/labels/isat.yaml")
            data["dataset"] = ds

        # 2) Auto map paths logic
        paths = data.get("paths", {})
        if isinstance(paths, dict):
            base_out = paths.get("base_output_dir")
            if base_out:
                root = f"{base_out}/{track}/{exp_id}"
                paths.setdefault("baseline_root", root)
                paths.setdefault("coco_dir", f"{root}/coco")
                paths.setdefault("dataset_view_dir", f"{root}/dataset_view")
                paths.setdefault("dataset_view_images_dir", f"{root}/dataset_view/images")
                paths.setdefault("dataset_view_coco_dir", f"{root}/dataset_view/cocoOut")
                paths.setdefault("train_output_dir", f"{root}/train")
                paths.setdefault("logs_dir", f"{root}/logs")
            data["paths"] = paths

        # 3) Auto map artifacts logic
        artifacts = data.get("artifacts", {})
        if isinstance(artifacts, dict) and "baseline_root" in data.get("paths", {}):
            root = data["paths"]["baseline_root"]
            artifacts.setdefault("run_manifest", f"{root}/run_manifest.json")
            artifacts.setdefault("split_summary", f"{root}/split_summary.json")
            artifacts.setdefault("train_args", f"{root}/train_args.json")
            artifacts.setdefault("eval_metrics", f"{root}/eval_metrics.json")
            artifacts.setdefault("artifact_index", f"{root}/artifact_index.json")
            data["artifacts"] = artifacts

        return data

    @classmethod
    def load(cls, config_path: str | Path) -> 'ExperimentConfig':
        import json
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def save(self, config_path: str | Path) -> None:
        import json
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2)

from __future__ import annotations

from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Dict, Any
from pathlib import Path

class DatasetConfig(BaseModel):
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


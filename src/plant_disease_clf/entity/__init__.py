import os, sys
from pathlib import Path

from dataclasses import dataclass


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    kaggle_URL: Path
    kaggle_file: Path


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_model_path: Path
    INCLUDE_TOP: bool
    INPUT_SHAPE: list
    WEIGHTS: str
    CLASSES: int
    OPTIMIZER: str
    LOSS: str
    METRICS: list


@dataclass(frozen=True)
class PrepareCallbacks:
    root_dir: Path
    tensorboard_logs: Path
    model_checkpoint: Path


@dataclass(frozen=True)
class Training:
    root_dir: Path
    model_path: Path
    model_metrics_path: Path
    data: Path
    updated_model_path: Path
    INPUT_SHAPE: list
    BATCH_SIZE: int
    SHUFFLE: bool
    VALIDATION_SPLIT: float
    LABEL_MODEL: str
    EPOCHS: int
    AUGMENTED: bool

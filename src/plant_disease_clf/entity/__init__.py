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

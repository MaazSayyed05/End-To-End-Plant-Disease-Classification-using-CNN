
import os,sys
from pathlib import Path

from dataclasses import dataclass


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    kaggle_URL: Path
    kaggle_file: Path





import os, sys
from pathlib import Path

from plant_disease_clf.utils.common import create_directories, read_yaml
from plant_disease_clf.constants import *

from plant_disease_clf.logger import logging
from plant_disease_clf.exception import CustomException

from plant_disease_clf.entity import DataIngestionConfig


class ConfigManager:
    def __init__(
        self, config_file_path=CONFIG_FILE_PATH, params_file_path=PARAMS_FILE_PATH
    ):

        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:

        self.config = self.config.data_ingestion
        return DataIngestionConfig(
            root_dir=self.config.root_dir, kaggle_URL=self.config.kaggle_URL, kaggle_file = self.config.kaggle_file
        )

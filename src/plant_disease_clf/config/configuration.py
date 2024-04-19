import os, sys
from pathlib import Path

from plant_disease_clf.utils.common import create_directories, read_yaml
from plant_disease_clf.constants import *

from plant_disease_clf.logger import logging
from plant_disease_clf.exception import CustomException

from plant_disease_clf.entity import DataIngestionConfig
from plant_disease_clf.entity import PrepareBaseModelConfig
from plant_disease_clf.entity import PrepareCallbacks


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
            root_dir=self.config.root_dir,
            kaggle_URL=self.config.kaggle_URL,
            kaggle_file=self.config.kaggle_file,
        )

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:

        self.config = self.config.base_model
        self.params = self.params.resnet50

        create_directories([self.config.root_dir])

        return PrepareBaseModelConfig(
            root_dir=self.config.root_dir,
            base_model_path=self.config.base_model_path,
            updated_model_path=self.config.updated_model_path,
            INCLUDE_TOP=self.params.INCLUDE_TOP,
            INPUT_SHAPE=self.params.INPUT_SHAPE,
            WEIGHTS=self.params.WEIGHTS,
            CLASSES=self.params.CLASSES,
            LOSS=self.params.LOSS,
            METRICS=self.params.METRICS,
            OPTIMIZER=self.params.OPTIMIZER,
        )

    def get_prepare_callbacks_config(self):
        self.config = self.config.callbacks
        create_directories([self.config.root_dir])

        return PrepareCallbacks(
            root_dir=self.config.root_dir,
            tensorboard_logs=self.config.tensorboard_logs,
            model_checkpoint=self.config.model_checkpoint,
        )

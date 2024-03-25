import os, sys
from pathlib import Path

from plant_disease_clf.logger import logging
from plant_disease_clf.exception import CustomException

# from plant_disease_clf.entity import DataIngestionConfig
from plant_disease_clf.config.configuration import ConfigManager
from plant_disease_clf.components.prepare_base_model import PrepareBaseModel


class PrepareBaseModelPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config_manager = ConfigManager()
            config = config_manager.get_prepare_base_model_config()
            base_model = PrepareBaseModel(config=config)
            base_model.updated_base_model()

        except Exception as e:
            raise CustomException(e, sys)

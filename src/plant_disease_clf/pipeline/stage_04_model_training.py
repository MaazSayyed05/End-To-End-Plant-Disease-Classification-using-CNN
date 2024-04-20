
import os,sys

from plant_disease_clf.logger import  logging
from plant_disease_clf.exception import  CustomException

from plant_disease_clf.config.configuration import ConfigManager
from plant_disease_clf.components.model_training import ModelTraining


class TrainingPipeline:
    def __init__(self):
        pass

    def main(self) -> list:
        try:
            config_manager = ConfigManager()
            config = config_manager.get_model_training_config()
            model_training = ModelTraining(config)
            model_training.train()

        except Exception as e:
            raise CustomException(e,sys)
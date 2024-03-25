import os, sys
from pathlib import Path

from plant_disease_clf.logger import logging
from plant_disease_clf.exception import CustomException

# from plant_disease_clf.entity import DataIngestionConfig
from plant_disease_clf.config.configuration import ConfigManager
from plant_disease_clf.components.data_ingestion import DataIngestion



class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config_manager = ConfigManager()
            config = config_manager.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=config)
            data_ingestion.get_data()
            
        except Exception as e:
            raise CustomException(e, sys)

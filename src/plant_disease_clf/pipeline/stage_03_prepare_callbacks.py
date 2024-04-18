
import os,sys

from plant_disease_clf.logger import  logging
from plant_disease_clf.exception import  CustomException

from plant_disease_clf.config.configuration import ConfigManager
from plant_disease_clf.components.prepare_callbacks import Callbacks


class CallbacksPipeline:
    def __init__(self):
        pass

    def main(self) -> list:
        try:
            config_manager = ConfigManager()
            prepare_callbacks = config_manager.get_prepare_callbacks_config()
            callbacks = Callbacks(prepare_callbacks)
            callbacks_list = callbacks.get_callbacks_list()
            logging.info("Callbacks are created successfully")

            return callbacks_list

        except Exception as e:
            raise CustomException(e,sys)
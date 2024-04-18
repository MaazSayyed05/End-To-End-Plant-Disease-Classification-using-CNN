import os, sys
from plant_disease_clf.logger import logging
from plant_disease_clf.exception import CustomException
from plant_disease_clf.utils.common import create_directories

import tensorflow as tf

from plant_disease_clf.entity import PrepareCallbacks


class Callbacks:
    def __init__(self, config: PrepareCallbacks):
        self.config = config

    def get_tb_callback(self):
        return tf.keras.callbacks.TensorBoard(
            log_dir=self.config.tensorboard_logs, histogram_freq=1
        )

    def get_model_ckpt_callback(self):
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=self.config.model_checkpoint, save_best_only=True
        )

    def get_callbacks_list(self):
        return [self.get_tb_callback(), self.get_model_ckpt_callback()]

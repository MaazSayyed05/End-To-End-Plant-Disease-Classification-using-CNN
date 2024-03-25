import os, sys
from pathlib import Path

from plant_disease_clf.logger import logging
from plant_disease_clf.exception import CustomException

import tensorflow as tf

import keras
from keras.layers import Conv2D, Flatten, Dense, Input, Lambda, Dropout
from keras.models import Sequential, Model
from keras.applications.resnet50 import ResNet50

from keras.saving import load_model

from plant_disease_clf.entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def base_model(self):  ## ResNet50 model with imagenet weights
        try:
            model = ResNet50(
                include_top=self.config.INCLUDE_TOP,
                input_shape=self.config.INPUT_SHAPE,
                weights=self.config.WEIGHTS,
            )

            for layer in model.layers:
                layer.trainable = False

            model.summary()

            model.save(self.config.base_model_path)

            return model

        except Exception as e:
            raise CustomException(e, sys)

    def updated_base_model(self):  ## Top layers will be added here
        try:

            self.base_model = self.base_model()

            self.full_model = Sequential()
            self.full_model.add(self.base_model)
            self.full_model.add(Flatten())

            self.full_model.add(Dense(256, activation="relu"))
            self.full_model.add(Dropout(0.4))
            self.full_model.add(Dense(self.config.CLASSES, activation="softmax"))

            self.full_model.summary()

            self.full_model.compile(
                optimizer=self.config.OPTIMIZER,
                loss=self.config.LOSS,
                metrics=self.config.METRICS,
            )

            self.full_model.save(self.config.updated_model_path)

            return self.full_model

        except Exception as e:
            raise CustomException(e, sys)

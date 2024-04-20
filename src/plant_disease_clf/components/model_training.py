import os, sys
from pathlib import Path
from plant_disease_clf.logger import logging
from plant_disease_clf.exception import CustomException

from plant_disease_clf.utils.common import save_json

import tensorflow as tf
from plant_disease_clf.pipeline.stage_03_prepare_callbacks import CallbacksPipeline

from plant_disease_clf.entity import Training


class ModelTraining:
    def __init__(self, config: Training):

        self.config = config

    def get_updated_base_model(self):
        self.model = tf.keras.models.load_model(self.config.updated_model_path)

    def train_valid_generator(self):

        data_generator_kwargs = dict(
            rescale=1.0 / 255, validation_split=self.config.VALIDATION_SPLIT
        )

        # test_data_generator_kwargs = dict(
        #     rescale = 1./255,
        #     validation_split = 0.20
        # )

        data_flow_kwargs = dict(
            target_size=self.config.INPUT_SHAPE[:-1],
            batch_size=self.config.BATCH_SIZE,
            # shuffle = True,
            interpolation="bilinear",
        )

        valid_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            **data_generator_kwargs
        )

        self.valid_generator = valid_data_generator.flow_from_directory(
            directory=self.config.data,
            subset="validation",
            shuffle=False,
            class_mode=self.config.LABEL_MODEL,
            **data_flow_kwargs
        )

        # test_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        #     **test_data_generator_kwargs
        # )

        # self.test_generator = test_data_generator.flow_from_directory(
        #     data = self.valid_generator,
        #     subset = 'validation',
        #     shuffle = False,
        #     class_mode='categorical',
        #     **data_flow_kwargs
        # )

        if self.config.AUGMENTED:
            train_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **data_generator_kwargs
            )

        else:
            train_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
                **data_generator_kwargs
            )

        self.train_generator = train_data_generator.flow_from_directory(
            directory=self.config.data,
            subset="training",
            shuffle=True,
            class_mode=self.config.LABEL_MODEL,
            **data_flow_kwargs
        )

    def get_data(self):
        self.train_data, self.val_data = tf.keras.utils.image_dataset_from_directory(
            directory=self.config.data,
            labels="inferred",
            label_mode=self.config.LABEL_MODEL,
            batch_size=self.config.BATCH_SIZE,
            image_size=self.config.INPUT_SHAPE[:-1],
            shuffle=self.config.SHUFFLE,
            seed=42,  # --------------------- add this for splitting the data
            validation_split=self.config.VALIDATION_SPLIT,
            subset="both",
            interpolation="bilinear",
        )
        self.train_data_samples = tf.data.experimental.cardinality(
            self.train_data
        ).numpy()
        self.val_data_samples = tf.data.experimental.cardinality(self.val_data).numpy()

        # print(self.train_data['labels'][0], self.val_data['labels'][0])
        print(self.train_data.class_names)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self):
        self.steps_per_epoch = self.train_data_samples // self.config.BATCH_SIZE

        self.validation_steps = self.val_data_samples // self.config.BATCH_SIZE

        callbacks = CallbacksPipeline()
        callbacks_list = callbacks.main()

        history = self.model.fit(
            self.train_data,
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.config.EPOCHS,
            validation_data=self.val_data,
            validation_steps=self.validation_steps,
            callbacks=callbacks_list,
        )

        save_json(self.config.model_metrics_path, history.history)

        self.save_model(
            path = self.config.model_path,
            model =self.model
        )

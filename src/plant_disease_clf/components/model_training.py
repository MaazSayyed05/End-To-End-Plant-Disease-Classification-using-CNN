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

    def get_updated_model(self):
        return tf.keras.models.load_model(self.config.updated_model_path)

    def train(self):
        try:
            logging.info(f"Training the model with the following config: {self.config}")

            callbacks = CallbacksPipeline()
            callbacks_list = callbacks.main()

            # -------------------------------------------------------------------------------------------------------------------------------------
            # train_data, val_data = tf.keras.utils.image_dataset_from_directory(
            #     directory = self.config.data,
            #     labels = "inferred",
            #     label_mode = self.config.LABEL_MODEL,
            #     batch_size = self.config.BATCH_SIZE,
            #     image_size = self.config.INPUT_SHAPE,
            #     shuffle = self.config.SHUFFLE,
            #     seed = 42,  #--------------------- add this for splitting the data
            #     validation_split = self.config.VALIDATION_SPLIT,
            #     subset = self.config.SUBSET,
            #     interpolation = "bilinear",
            # )
            # train_data_samples = tf.data.experimental.cardinality(train_data).numpy()
            # val_data_samples = tf.data.experimental.cardinality(val_data).numpy()
            # -------------------------------------------------------------------------------------------------------------------------------------
            data_generator_kwargs = dict(rescale=1.0 / 255, validation_split=0.20)

            data_flow_kwargs = dict(
                target_size=self.config.INPUT_SHAPE[:-1],
                batch_size=self.config.BATCH_SIZE,
                interpolation="bilinear",
            )

            valid_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                **data_generator_kwargs
            )

            val_data = valid_data_gen.flow_from_directory(
                directory=self.config.data,
                class_mode=self.config.LABEL_MODEL,
                shuffle=self.config.SHUFFLE,
                seed=42,  # --------------------- add this for splitting the data
                subset="validation",
                # interpolation = "bilinear",
                **data_flow_kwargs,
            )

            if self.config.AUGMENTED:
                train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                    rotation_range=40,
                    horizontal_flip=True,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2,
                    **data_generator_kwargs,
                )

            else:
                train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                    **data_generator_kwargs
                )

            train_data = train_data_gen.flow_from_directory(
                directory=self.config.data,
                class_mode=self.config.LABEL_MODEL,
                shuffle=True,  ###
                # seed = 42,  #--------------------- add this for splitting the data
                subset="training",
                # interpolation = "bilinear",
                **data_flow_kwargs,
            )

            model = self.get_updated_model()

            print(callbacks_list)
            print(model.summary())

            # history = model.fit(
            #     train_data,
            #     epochs = self.config.EPOCHS,
            #     validation_data = val_data,
            #     callbacks = callbacks_list
            # )

            self.steps_per_epoch = train_data.samples // self.config.BATCH_SIZE

            self.validation_steps = val_data.samples // self.config.BATCH_SIZE

            history = model.fit(
                train_data,
                steps_per_epoch=self.steps_per_epoch,
                epochs=self.config.EPOCHS,
                validation_data=val_data,
                validation_steps=self.validation_steps,
                callbacks=callbacks_list,
            )

            # tf.keras.models.save_model(model, self.config.model_path)
            # save_json(self.config.model_metrics_path, history.history)
            logging.info("Model training completed successfully")

        except Exception as e:
            logging.error(f"Model training failed: {e}")
            raise CustomException(e, sys)

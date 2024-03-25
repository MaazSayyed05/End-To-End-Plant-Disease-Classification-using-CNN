import os, sys

from plant_disease_clf.logger import logging
from plant_disease_clf.exception import CustomException

from plant_disease_clf.pipeline.stage_01_data_ingestion import (
    DataIngestionTrainingPipeline,
)
from plant_disease_clf.pipeline.stage_02_prepare_base_model import (
    PrepareBaseModelPipeline,
)


STAGE_NAME = "Data Ingestion Stage"

try:
    logging.info(f">>>>>>>> {STAGE_NAME} Started <<<<<<<")
    # data_ingestion = DataIngestionTrainingPipeline()
    # data_ingestion.main()
    logging.info(f">>>>>>>> {STAGE_NAME} Complted <<<<<<<")

except Exception as e:
    logging.error(f">>>>>>>> {STAGE_NAME} Failed <<<<<<<")
    raise e


STAGE_NAME = "Prepare Base Model Stage"

try:
    logging.info(f">>>>>>>> {STAGE_NAME} Started <<<<<<<")
    base_model = PrepareBaseModelPipeline()
    base_model.main()
    logging.info(f">>>>>>>> {STAGE_NAME} Complted <<<<<<<")

except Exception as e:
    logging.error(f">>>>>>>> {STAGE_NAME} Failed <<<<<<<")
    raise e
